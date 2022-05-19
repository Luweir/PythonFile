# import pymysql

from config import appid, token
import qqbot
#
# db = pymysql.connect(host="localhost", user="root", password="1234", database="book")
# cursor = db.cursor()

token = qqbot.Token(appid, token)
api = qqbot.UserAPI(token, False)
user = api.me()

print(user.username)  # 打印机器人名字


# 消息处理
def _message_handler(event, message: qqbot.Message):
    msg_api = qqbot.MessageAPI(token, False)
    # 打印日志中的返回信息
    message_content = message.content
    message_content = message_content[message_content.find('>') + 2:]
    qqbot.logger.info("event %s" % event + ",receive message %s" % message.content)
    print("message author:", message.author.id, message.author.username, message.author.bot)
    if message_content == "打卡规则":
        send = qqbot.MessageSendRequest("<@%s>打卡规则：......" % message.author.id, message.id)
    elif message_content == "喝水打卡":
        send = qqbot.MessageSendRequest("<@%s>喝水打卡成功！请继续保持！" % message.author.id, message.id)
    elif message_content == "运动打卡":
        send = qqbot.MessageSendRequest("<@%s>运动打卡成功！请继续保持！" % message.author.id, message.id)
    elif message_content == "视力打卡":
        send = qqbot.MessageSendRequest("<@%s>视力打卡成功！请继续保持！" % message.author.id, message.id)
    elif message_content == "午睡打卡":
        send = qqbot.MessageSendRequest("<@%s>运动打卡成功！请继续保持！" % message.author.id, message.id)
    elif message_content == "查询状态":
        send = qqbot.MessageSendRequest("<@%s>您的状态为：xxxx" % message.author.id, message.id)
    else:
        send = qqbot.MessageSendRequest("<@%s>谢谢你，加油" % message.author.id, message.id)
    # 通过api发送回复消息
    msg_api.post_message(message.channel_id, send)


# 注册事件类型和回调，可以注册多个
qqbot_handler = qqbot.Handler(qqbot.HandlerType.AT_MESSAGE_EVENT_HANDLER, _message_handler)
qqbot.listen_events(token, False, qqbot_handler)
