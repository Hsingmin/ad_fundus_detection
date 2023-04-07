
import csv, predict, os

def run_model(file_name):
    # predicts
    print("processing file {}".format(file_name))
    result = predict.predict(file_name)
    print("result: {}".format(result))

# tests interface
if __name__ == '__main__':
    run_model("13324_right.jpeg")
