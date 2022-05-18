import pymysql

db = pymysql.connect(host="localhost", user="root", password="1234", database="test")
cursor = db.cursor()
cursor.execute("select * from clock_in where id=1")
results = cursor.fetchall()
print(results)
print(len(results))

