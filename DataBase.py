from dotenv import load_dotenv
import os
import psycopg2


class DBConnector:
    def __init__(self):
        load_dotenv()  # Load variables from the .env file
        self.user = os.getenv('DB_USER')
        self.password = os.getenv('DB_PASSWORD')
        self.host = os.getenv('DB_HOST')
        self.port = os.getenv('DB_PORT')
        self.conn = None
        self.db_exist = False

    def connect_to_db(self, db_name: str):
        try:
            # Connect to the PostgreSQL database

            if db_name:
                print(f"Connecting To DataBase {db_name}")
                conn = psycopg2.connect(
                    dbname=db_name,
                    user=self.user,
                    password=self.password,
                    host=self.host,
                    port=self.port
                )
                conn.autocommit = True

                self.conn = conn
                print(f"Connection To DataBase {db_name}, Successful")
                return self.conn
            else:
                print(f"Connecting To DataBase Instance")
                conn = psycopg2.connect(
                    dbname='postgres',
                    user=self.user,
                    password=self.password,
                    host=self.host,
                    port=self.port
                )
                conn.autocommit = True
                self.conn = conn
                print("Connection To DataBase Successful")
                return self.conn
        except psycopg2.Error as e:
            print("Error connecting to PostgreSQL:", e)

    def check_db(self, db_name):
        # Create a cursor object to execute SQL commands
        print("Checking if DataBase Exist")
        cur = self.conn.cursor()

        # Execute a query to fetch the list of databases
        cur.execute("SELECT datname FROM pg_database;")

        # Fetch all rows from the result set
        rows = cur.fetchall()

        # Extract the values from the fetched rows
        database_list = [row[0] for row in rows]
        self.db_exist = db_name in database_list

        # Close the cursor and connection
        cur.close()
        return self.db_exist

    def create_db(self, db_name):
        # Create a cursor object to execute SQL commands
        print("Creating Database")
        cur = self.conn.cursor()

        # Execute a SQL command to create a new database
        cur.execute(f'CREATE DATABASE {db_name};')

        # Commit the transaction to apply changes
        self.conn.commit()
        print("Created Database Successfully")
        cur.close()
        self.conn.close()

    def close_db_connection(self):
        # Close the cursor and connection
        # cur.close()
        print("Closing Connection To DB")
        self.conn.close()
        print("Connection To DB Closed")

    def drop_bd(self, db_name):
        # Create a cursor object to execute SQL commands
        print("Dropping DataBase")
        cur = self.conn.cursor()

        # Execute a SQL command to create a new database
        cur.execute(f"DROP DATABASE {db_name};")

        # Commit the transaction to apply changes
        self.conn.commit()
        print("Successfully Dropped DataBase")
        cur.close()

    def main_connection_to_db(self, db_name):
        db_name = db_name.lower()
        try:
            self.connect_to_db('')
            if self.check_db(db_name):
                self.connect_to_db(db_name)
            else:
                self.create_db(db_name)
                self.connect_to_db(db_name)

            return self.conn
        except Exception as E:
            print(E)

    def create_message_table(self):
        # Create a cursor object to execute SQL commands
        print("Creating Message Table")
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS Messages (
        message_id SERIAL PRIMARY KEY,
        message_text TEXT,
        sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Commit the transaction to apply changes
        self.conn.commit()
        print("Created Message Table Successfully")
        cur.close()

    def create_response_table(self):
        # Create a cursor object to execute SQL commands
        print("Creating Responses Table")
        cur = self.conn.cursor()
        # Execute a SQL command to create a new database
        cur.execute("""
        CREATE TABLE IF NOT EXISTS Responses (
        response_id SERIAL PRIMARY KEY,
        response_text TEXT,
        message_id INT NOT NULL,
        responded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (message_id) REFERENCES Messages(message_id)
        )
        """)

        # Commit the transaction to apply changes
        self.conn.commit()
        print("Created Responses Table Successfully")
        cur.close()

    def create_todos_table(self):
        # Create a cursor object to execute SQL commands
        print("Creating Todos Table")
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS Todos (
        todo_id SERIAL PRIMARY KEY,
        to_do_name TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Commit the transaction to apply changes
        self.conn.commit()
        print("Created Todos Table Successfully")
        cur.close()

    def create_todo_item_table(self):
        # Create a cursor object to execute SQL commands
        print("Creating Todos Item Table")
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS TodosItem (
        todo_item_id SERIAL PRIMARY KEY,
        task TEXT,
        todo_id INT NOT NULL,
        done BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (todo_id) REFERENCES Todos(todo_id)
        )
        """)

        # Commit the transaction to apply changes
        self.conn.commit()
        print("Created Todos Item Table Successfully")
        cur.close()

    def create_message_record(self, message_data: str):
        # Create a cursor object to execute SQL commands
        cur = self.conn.cursor()

        # Insert data into Messages table
        insert_into_messages = '''
            INSERT INTO Messages (message_text)
            VALUES (%s)
        '''
        message_data = (message_data,)
        cur.execute(insert_into_messages, message_data)
        # Commit the changes to the database
        self.conn.commit()
        # Close the cursor
        cur.execute("""
        SELECT * 
        FROM messages 
        ORDER BY sent_at DESC 
        LIMIT 1
        """)
        inserted_data = cur.fetchone()
        cur.close()
        return inserted_data if inserted_data else None

    def create_todo_record(self, todo_name: str):
        # Create a cursor object to execute SQL commands
        cur = self.conn.cursor()

        # Insert data into Messages table
        insert_into_todos = '''
            INSERT INTO Todos (to_do_name)
            VALUES (%s)
        '''
        todo_data = (todo_name,)
        cur.execute(insert_into_todos, todo_data)
        # Commit the changes to the database
        self.conn.commit()
        # Close the cursor
        cur.execute("""
        SELECT * 
        FROM Todos 
        ORDER BY created_at DESC 
        LIMIT 1
        """)
        inserted_data = cur.fetchone()
        cur.close()
        return inserted_data if inserted_data else None

    def create_response_record(self, response_data: str, message_id: int):
        # Create a cursor object to execute SQL commands
        cur = self.conn.cursor()

        # Insert data into Responses table
        insert_into_responses = '''
            INSERT INTO Responses (message_id, response_text)
            VALUES (%s, %s)
        '''
        response_data = (message_id, response_data)
        cur.execute(insert_into_responses, response_data)
        # inserted_data = cur.fetchone()
        # print(inserted_data)
        # # Close the cursor and connection
        cur.close()
        return True

    def create_todo_item_record(self, response_data: str, todo_id: int):
        # Create a cursor object to execute SQL commands
        cur = self.conn.cursor()

        # Insert data into Responses table
        insert_into_todo_items = '''
            INSERT INTO TodosItem (todo_id, task)
            VALUES (%s, %s)
        '''
        response_data = (todo_id, response_data)
        cur.execute(insert_into_todo_items, response_data)
        # inserted_data = cur.fetchone()
        # print(inserted_data)
        # # Close the cursor and connection
        cur.close()
        return True

    def update_todo_item_record(self, response_data: str, todo_id: int, done: int):
        # Create a cursor object to execute SQL commands
        cur = self.conn.cursor()

        # Insert data into Responses table
        # Update query
        update_todo_item = '''
            UPDATE TodosItem
            SET task = %s, done = %s
            WHERE todo_id = %s
        '''
        response_data = (response_data, done, todo_id)
        cur.execute(update_todo_item, response_data)

        cur.close()
        return True

    def run_customer_query(self, query):
        # Create a cursor object to execute SQL commands
        cur = self.conn.cursor()
        cur.execute(query)

        inserted_data = cur.fetchall()
        cur.close()
        return inserted_data if inserted_data else None
