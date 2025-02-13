import a_Data_ingestion 


def main():
    path="Dataset/isic-2024-challenge/train-metadata.csv"
    train= a_Data_ingestion.read_data(path)
    print("Data Ingestion Completed")