import pymysql
DB_HOST = "localhost"
DB_USER = "myuser118"
DB_PASSWORD = "1234"
DB_NAME = "mydb118"

def make_sent_table():
    rand_sent = ["안녕하세요", "감사해요", "잘 있어요", "다시 만나요"]

    def all_clear_train_data(db):
        sql = '''
                delete from rand_sent
            '''
        with db.cursor() as cursor:
            cursor.execute(sql)

        sql = '''
        ALTER TABLE rand_sent AUTO_INCREMENT=1
        '''
        with db.cursor() as cursor:
            cursor.execute(sql)

    def insert_data(db, data):
        sql = '''INSERT rand_sent(sent) VALUES('%s')''' % (data)

        sql = sql.replace("'None'", "null")

        with db.cursor() as cursor:
            cursor.execute(sql)
            print('저장')
            db.commit()

    db = None
    try:
        db = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            passwd=DB_PASSWORD,
            db=DB_NAME,
            charset='utf8'
        )
        
        sql = '''
            CREATE TABLE IF NOT EXISTS `rand_sent` (
                `sent` VARCHAR(30) NOT NULL UNIQUE
            )
            ENGINE = InnoDB DEFAULT CHARSET=utf8
        '''

        with db.cursor() as cursor:
            cursor.execute(sql)
            
        print('랜덤 문장 테이블 생성 성공')

        all_clear_train_data(db)

        for i in rand_sent:
            insert_data(db, i)

    except Exception as e:
        print(e)

    finally:
        if db is not None:
            db.close()

if __name__ == '__main__':
    make_sent_table()