"""
Thin shell wrapping connection to the lab database

Usage:
    db = LabDB()

    results = db.execute("SELECT * FROM experiments")
"""

#import MySQLdb
import sqlite3
import os

class LabDB:
    hostname = "taylor.physics.mun.ca"

    def __init__(self):

        hostname = os.uname()[1]
        if hostname == self.hostname:
            self.hostname = "localhost"

        # Open database connection
        self.conn = sqlite3.connect(self.hostname, "lab", "fluids0", "lab")

    def execute(self, sql):
        cursor = self.conn.cursor()

        nrows = cursor.execute(sql)

        result = cursor.fetchall()

        return result

    def execute_one(self, sql):
        cursor = self.conn.cursor()

        nrows = cursor.execute(sql)
        result = cursor.fetchone()

        return result

    def commit(self):
        self.conn.commit()
    def rollback(self):
        self.conn.rollback()
    def close(self):
        self.conn.close()


if __name__ == "__main__":
    db = LabDB()
    print db.execute("SELECT * FROM experiments")
    db.close()


