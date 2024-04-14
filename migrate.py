import argparse
from db.migrator import Migrator

parser = argparse.ArgumentParser(prog='Migrator', description='Migrations helper')
parser.add_argument('-s', '--status', action='store_true', help='get migrations status')
parser.add_argument('-p', '--pretend', action='store_true', help='print SQL\'s to console instead of executing')
parser.add_argument('-n', type=int, metavar='NUMBER_OF_MIGRATIONS',
                    help='specify number of migrations to migrate (default: 1)')
parser.add_argument('--date', metavar='MIGRATION_DATE', help='specify migration to migrate')

args = parser.parse_args()
migrator = Migrator()

if args.status:
    string = '{:^6} {}'
    print(string.format('Status', 'Migration'))
    migrations = migrator.getStatus(args.n if args.n is not None else 10)
    for mig in migrations:
        print(string.format('Y' if mig[0] else 'N', mig[1]))
elif args.date is not None:
    migrator.migrateByDate(args.date, args.pretend)
else:
    migrator.migrateNew(args.n, args.pretend)
