import os
import psycopg2
from dotenv import load_dotenv
from datetime import datetime


class Migrator:
    def __init__(self):
        load_dotenv()
        self.conn = psycopg2.connect(host=os.getenv('POSTGRES_HOST'), dbname=os.getenv('POSTGRES_DB'),
                                     user=os.getenv('POSTGRES_USER'), password=os.getenv('POSTGRES_PASSWORD'),
                                     port=os.getenv('POSTGRES_PORT'))
        self.cursor = self.conn.cursor()

    path = os.path.dirname(__file__) + '/../migrations/'
    status_count = 10

    def getLastMigrationsDB(self, n=10) -> list:
        self.cursor.execute(f"""
                           SELECT id, name, date FROM public.migrations ORDER BY "name" DESC LIMIT {n};
                           """)
        return self.cursor.fetchall()

    def getNewMigrationFiles(self):
        migrated = [mig[1] for mig in self.getLastMigrationsDB(100)]
        new_files = [file for file in os.listdir(self.path) if
                     os.path.isfile(self.path + file) and file.endswith(".sql") and file not in migrated]
        new_files.sort()
        return new_files

    def getStatus(self, n=0):
        result = []
        n = n if n > 0 else self.status_count
        new_files = self.getNewMigrationFiles()
        db_mig = self.getLastMigrationsDB(n if n > 0 else 10)
        db_mig.reverse()

        migrated = [(True, file[1]) for file in db_mig]
        result.extend(migrated)

        not_migrated = [(False, file) for file in new_files]
        result.extend(not_migrated)

        result.sort(key=lambda mig: mig[1])

        return result[-n:]

    def migrateNew(self, n=None, pretend=False):
        files_to_migrate = []
        new_files = self.getNewMigrationFiles()

        if n is not None:
            for i in range(0, min(n, len(new_files))):
                files_to_migrate.append(new_files[i])
        else:
            files_to_migrate = new_files

        return self.__migrate(files_to_migrate, pretend)

    def migrateByDate(self, date, pretend=False):
        new_files = self.getNewMigrationFiles()
        files_to_migrate = [file for file in new_files if date in file]

        if len(files_to_migrate) > 1:
            string = '{:^6} {}'
            print('Got more than 1 migration by that date:')
            print(string.format('Number', 'Migration'))
            for i, file in enumerate(files_to_migrate):
                print(string.format(i, file))
            nums = input('Specify numbers separated by (,) to migrate or [a]ll: ')
            nums_to_migrate = [num.strip() for num in nums.split(',')]
            if all(s not in nums_to_migrate for s in ['a', 'all']):
                files_to_migrate = [file for i, file in enumerate(files_to_migrate) if str(i) in nums_to_migrate]

        return self.__migrate(files_to_migrate, pretend)

    def __migrate(self, files_to_migrate, pretend=False):
        migrated = []
        for file in files_to_migrate:
            print('Migrating ' + file)

            if pretend:
                print(self.cursor.mogrify(open(self.path + file, 'r').read()).decode("utf-8"))
                continue

            try:
                self.cursor.execute(open(self.path + file, 'r').read())
            except psycopg2.errors.Error as e:
                print('Got an error while migrating ' + file)
                print(e)
                exit(1337)
            else:
                migrated.append((file, datetime.now().astimezone()))
                self.conn.commit()
                print('Migrated ' + file)
        self.cursor.executemany("""
        INSERT INTO public.migrations ("name", "date") VALUES (%s, %s);
        """, migrated)
        self.conn.commit()

        return migrated
