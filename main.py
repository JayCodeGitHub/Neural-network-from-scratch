import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from network import network


def main():
    network()

if __name__ == "__main__":
    main()
