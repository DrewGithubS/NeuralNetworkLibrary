from FinalNetwork import Network
from random import random
import csv, math


print("Reading data...")
data = []
with open('C:\\Users\\User\\Desktop\\FinalNetwork\\train.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    float_row = []
    for count, row in enumerate(readCSV):
        if(count == 500):
            break
        try:
            float_row.append(int(row[0]))
            for item in row[1:]:
                float_row.append(float(item)/255)
        except:
            pass
        data.append(float_row)


def check_correct(outputs, correct):
    highest_count = -1
    highest_value = 0
    for count, value in enumerate(outputs):
        if(value > highest_value):
            highest_count = count
            highest_value = value
    return highest_count == correct

def activation(num):
    return ((math.atan(num)/math.pi)+0.5)
def activation_inverse(num):
    return math.tan((num-1)*math.pi)
print("Training network...")
network = Network(784, [16, 16, 16], 10, 0.65, activation, activation_inverse)
correct = []
for count, piece in enumerate(data):
    if(count%1000 == 0):
        print(f"{count} images trained...")
        with open('C:\\Users\\User\\Desktop\\FinalNetwork\\cost.txt', 'w') as f:
            f.write(str(network.cost))
        with open('C:\\Users\\User\\Desktop\\FinalNetwork\\correct.txt', 'w') as f:
            f.write(str(correct))
    network.set_inputs(piece[1:])
    expected_output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    expected_output[piece[0]] = 1
    network.feed_forward()
    correct.append(check_correct(network.outputs, piece[0]))
    network.backpropogate(expected_output, False)
network.set_inputs(data[0][1:])
network.feed_forward()
expected_output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
expected_output[data[0][0]] = 1
network.backpropogate(expected_output, True)
network.save("C:\\Users\\User\\Desktop\\FinalNetwork")

with open('C:\\Users\\User\\Desktop\\FinalNetwork\\cost.txt', 'w') as f:
    f.write(str(network.cost))
with open('C:\\Users\\User\\Desktop\\FinalNetwork\\correct.txt', 'w') as f:
    f.write(str(correct))
