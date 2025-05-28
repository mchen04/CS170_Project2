import random

#rand function
def test_evaluate_subset(x):
    return random.uniform(0.0, 100.0)

#main ui
def main():
    print("Welcome to Tabito Sakamoto's Feature Selection Algorithm.")

    try:
        num_features = input("Please enter total number of features: ")
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return

    print("Type the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")

    choice = input()

    if choice == "1":
        print("Forward Selection will run here.")  
    elif choice == "2":
        print("Backward Elimination will run here.")  
    else:
        print("Invalid input. Please enter 1 or 2.")