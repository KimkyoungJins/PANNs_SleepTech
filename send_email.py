import smtplib
from email.mime.text import MIMEText
import sys

def send(subject, body):
    sender = "kkjin722@gmail.com"
    receiver = "kkjin722@gmail.com"
    app_password = "srnbotsplnwxspcw"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender, app_password)
        server.sendmail(sender, receiver, msg.as_string())
    print("Email sent: " + subject)

if __name__ == '__main__':
    subject = sys.argv[1] if len(sys.argv) > 1 else "학습 완료"
    body = sys.argv[2] if len(sys.argv) > 2 else "학습이 완료되었습니다."
    send(subject, body)
