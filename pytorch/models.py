import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank  # 오디오 → 스펙트로그램 변환 도구
from torchlibrosa.augmentation import SpecAugmentation  # 스펙트로그램 데이터 증강 도구

from pytorch_utils import do_mixup  # Mixup 증강 함수 (두 샘플을 섞어서 새 샘플 생성)


def init_layer(layer):
    """Linear 또는 Conv 레이어의 가중치를 Xavier 방식으로 초기화.
    Xavier 초기화: 입출력 크기에 맞게 적절한 범위로 가중치를 설정하여
    학습 초기에 그래디언트가 너무 크거나 작아지는 것을 방지.
    """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)  # 바이어스는 0으로 초기화


def init_bn(bn):
    """BatchNorm 레이어 초기화.
    weight=1, bias=0으로 설정하면 처음에는 입력을 그대로 통과시키는 것과 같음.
    """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    """컨볼루션 블록: Conv → BN → ReLU → Conv → BN → ReLU → Pooling

    스펙트로그램(2D 이미지)에서 패턴(특징)을 추출하는 기본 단위.
    3x3 컨볼루션을 2번 적용한 후 풀링으로 크기를 줄임.
    VGG 네트워크 스타일의 구조.
    """
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        # 첫 번째 3x3 컨볼루션: 입력 채널 → 출력 채널
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)  # padding=1이면 입출력 크기 동일

        # 두 번째 3x3 컨볼루션: 출력 채널 → 출력 채널 (채널 수 유지)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)

        # BatchNorm: 각 채널별로 평균=0, 분산=1로 정규화 → 학습 안정화
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)


    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        """
        Args:
            input: (batch, channels, height, width) 형태의 스펙트로그램
            pool_size: 풀링 크기. (2,2)면 가로세로 각각 절반으로 줄임
            pool_type: 'avg'(평균), 'max'(최대값), 'avg+max'(둘 다 합산)

        Returns:
            풀링된 특징맵 (batch, out_channels, height/2, width/2)
        """
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))  # Conv1 → BN → ReLU (음수를 0으로 만듦)
        x = F.relu_(self.bn2(self.conv2(x)))  # Conv2 → BN → ReLU
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)  # 영역에서 최대값만 선택
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)  # 영역의 평균값 사용
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2  # 평균 + 최대값을 합산 → 더 풍부한 정보
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn14_16k(nn.Module):
    """16kHz 오디오용 CNN14 모델 (수면 단계 분류용으로 수정됨).

    전체 흐름:
    raw waveform (1D 소리, 30초 = 480,000 샘플)
      → STFT (스펙트로그램으로 변환)
      → Log-Mel (사람 귀에 맞게 주파수 스케일 변환)
      → 6개의 ConvBlock (특징 추출, 점점 고수준 패턴 학습)
      → Global Pooling (시간 축 압축)
      → FC Layer (분류)
      → 3개 클래스 logits (REM, NREM, Wake)

    [변경사항 - 수면 분류용]
    - classes_num: 527 → 3 (REM, NREM, Wake)
    - 출력: sigmoid 제거 → raw logits 반환 (CrossEntropy 손실이 내부에서 softmax 처리)
    - 오디오 길이: 10초 → 30초 (구조 변경 없이 그대로 동작, Global Pooling 덕분)
    """
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num):

        super(Cnn14_16k, self).__init__()

        # ===== 파라미터 검증 (16kHz 모델은 반드시 이 값들을 사용해야 함) =====
        assert sample_rate == 16000  # 1초에 16,000개 샘플
        assert window_size == 512    # STFT 윈도우 크기 (한 번에 분석할 샘플 수)
        assert hop_size == 160       # STFT 이동 간격 (윈도우를 160샘플씩 이동)
        assert mel_bins == 64        # 멜 필터뱅크 수 (주파수를 64개 구간으로 나눔)
        assert fmin == 50            # 분석할 최소 주파수 (50Hz)
        assert fmax == 8000          # 분석할 최대 주파수 (8000Hz, 나이퀴스트 = 16000/2)

        # STFT 설정
        window = 'hann'       # 한 윈도우(Hann): 양 끝이 부드럽게 감소하는 창함수
        center = True         # 프레임 중심 기준으로 패딩
        pad_mode = 'reflect'  # 오디오 시작/끝에서 거울 반사 방식으로 패딩
        ref = 1.0             # 데시벨 변환 시 기준값
        amin = 1e-10          # log 계산 시 0 방지용 최소값
        top_db = None         # 최대 데시벨 제한 없음

        # ===== Step 1: Raw Waveform → Spectrogram (STFT) =====
        # 소리(1D) → 시간-주파수 2D 이미지로 변환
        # n_fft=512이면 주파수 bin 수 = 512/2+1 = 257개
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
            win_length=window_size, window=window, center=center, pad_mode=pad_mode,
            freeze_parameters=True)  # freeze: 이 부분은 학습하지 않음 (고정 연산)

        # ===== Step 2: Spectrogram → Log-Mel Spectrogram =====
        # 주파수 축을 사람 귀의 감도에 맞게 변환 (멜 스케일)
        # 257개 주파수 bin → 64개 mel bin으로 압축
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
            n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
            freeze_parameters=True)

        # ===== Step 3: SpecAugmentation (학습 시에만 적용) =====
        # 스펙트로그램의 일부를 랜덤으로 가려서(masking) 과적합 방지
        # time_drop: 시간 축에서 64프레임 폭으로 2번 마스킹
        # freq_drop: 주파수 축에서 8bin 폭으로 2번 마스킹
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)

        # ===== BatchNorm: mel_bins(64) 채널에 대해 정규화 =====
        self.bn0 = nn.BatchNorm2d(64)

        # ===== 6개의 ConvBlock: 점점 더 복잡한 패턴을 학습 =====
        # 채널 수: 1 → 64 → 128 → 256 → 512 → 1024 → 2048
        # 각 블록마다 (2,2) 풀링으로 크기가 절반씩 줄어듦
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)     # 저수준: 간단한 주파수 패턴
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)   # 약간 복잡한 패턴
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)  # 중간 수준 패턴
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)  # 고수준 패턴
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024) # 더 복잡한 패턴
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)# 가장 추상적인 패턴

        # ===== 분류를 위한 Fully Connected 레이어 =====
        self.fc1 = nn.Linear(2048, 2048, bias=True)                 # 2048차원 임베딩 생성
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)   # 최종 분류 (2048 → 3)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """모델의 순전파 (입력 → 출력).

        Args:
            input: (batch_size, data_length) 형태의 raw waveform
                   예: (16, 480000) = 16개의 30초 오디오 (16kHz)
            mixup_lambda: Mixup 증강 비율. 학습 시에만 사용, 추론 시 None

        Returns:
            output_dict: {
                'clipwise_output': (batch_size, 3) - 각 클래스의 logits (raw 점수),
                    F.cross_entropy가 내부에서 softmax를 적용하므로 여기서는 raw logits 반환.
                    추론 시 확률이 필요하면: torch.softmax(output['clipwise_output'], dim=1)
                'embedding': (batch_size, 2048) - 오디오의 특징 벡터
            }
        """

        # [Step 1] Raw waveform → 스펙트로그램 (STFT 적용)
        # 30초 오디오: (batch, 480000) → (batch, 1, 3001, 257)
        # time_steps = 480000 / 160 + 1 = 3001
        x = self.spectrogram_extractor(input)

        # [Step 2] 스펙트로그램 → Log-Mel 스펙트로그램
        # (batch, 1, 3001, 257) → (batch, 1, 3001, 64)
        # 257개 주파수 bin이 64개 mel bin으로 압축됨
        x = self.logmel_extractor(x)

        # [Step 3] BatchNorm 적용
        # BN은 채널 축(dim=1)에 적용되는데, 현재 mel_bins가 dim=3에 있으므로
        # transpose로 dim=1과 dim=3을 교환 → BN 적용 → 다시 원래 순서로
        x = x.transpose(1, 3)  # (batch, 64, 3001, 1)
        x = self.bn0(x)        # 64개 mel bin 각각을 정규화
        x = x.transpose(1, 3)  # (batch, 1, 3001, 64) 로 복원

        # [Step 4] SpecAugmentation (학습 시에만)
        # 스펙트로그램의 일부를 랜덤으로 0으로 채움 → 과적합 방지
        if self.training:
            x = self.spec_augmenter(x)

        # [Step 5] Mixup 증강 (학습 시에만)
        # 두 개의 서로 다른 오디오 스펙트로그램을 비율에 따라 섞음
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        # [Step 6] 6개의 ConvBlock 통과 (특징 추출)
        # 각 블록: Conv→BN→ReLU→Conv→BN→ReLU→AvgPool(2,2)
        # 30초 오디오 기준 shape 변화:
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')   # → (batch, 64, 1500, 32)
        x = F.dropout(x, p=0.2, training=self.training)  # 20% 뉴런을 랜덤으로 끔 (과적합 방지)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')   # → (batch, 128, 750, 16)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')   # → (batch, 256, 375, 8)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')   # → (batch, 512, 187, 4)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')   # → (batch, 1024, 93, 2)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')   # → (batch, 2048, 93, 2) 풀링 없음
        x = F.dropout(x, p=0.2, training=self.training)

        # [Step 7] Global Pooling - 시간과 주파수 축을 하나의 벡터로 압축
        # 이 단계 덕분에 입력 오디오 길이가 달라도 출력 크기는 항상 (batch, 2048)
        x = torch.mean(x, dim=3)  # 주파수 축 평균 → (batch, 2048, 93)

        (x1, _) = torch.max(x, dim=2)  # 시간 축 최대값 → (batch, 2048)
        x2 = torch.mean(x, dim=2)       # 시간 축 평균값 → (batch, 2048)
        x = x1 + x2  # 최대값 + 평균값을 합산 → 두 가지 관점의 정보를 모두 활용

        # [Step 8] 분류 레이어
        x = F.dropout(x, p=0.5, training=self.training)  # 50% 드롭아웃 (강한 정규화)
        x = F.relu_(self.fc1(x))          # FC: 2048 → 2048 + ReLU
        embedding = F.dropout(x, p=0.5, training=self.training)  # 이것이 오디오 임베딩 벡터

        # [변경] sigmoid 제거 → raw logits 반환
        # F.cross_entropy()가 내부에서 softmax를 자동으로 적용하므로,
        # 여기서는 sigmoid/softmax 없이 그대로 반환하는 것이 수치적으로 더 안정적
        clipwise_output = self.fc_audioset(x)  # FC: 2048 → 3 (raw logits)

        output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}

        return output_dict
