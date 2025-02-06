import argparse
from main import main as run_main


def parse_arguments():
    parser = argparse.ArgumentParser(description="Ответ на запрос из PDF и CSV документов.")
    parser.add_argument("file_paths", type=str, nargs='+', help="Пути к файлам (PDF и CSV).")
    parser.add_argument("query", type=str, help="Запрос пользователя.")
    args = parser.parse_args()


    args.query = args.query.replace("\\n", "\n").replace("\\t", "\t")
    return args


def main():
    args = parse_arguments()
    run_main(args.file_paths, args.query)


if __name__ == "__main__":
    main()
