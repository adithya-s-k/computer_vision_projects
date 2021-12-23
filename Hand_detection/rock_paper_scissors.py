import random

def gameOutPut(inputGesture):
    listOfGestures = ["rock", "paper", "scissor"]
    randomeGesture = random.randint(0,2)
    ComputerGesture = listOfGestures[int(randomeGesture)]
    
    if ComputerGesture == "rock":
        
        if inputGesture == "rock":
            return "Both Selected ",ComputerGesture," Its A Draw"
        elif inputGesture == "paper":
            return "You Selected ",inputGesture," and computer selected ",ComputerGesture,"So You Win "
        elif inputGesture == "scissor":
            return "You Selected ",inputGesture," and computer selected ",ComputerGesture,"So You Lose "
        
    elif ComputerGesture == "paper":
        
        if inputGesture == "paper":
            return "Both Selected ",ComputerGesture," Its A Draw"
        elif inputGesture == "scissor":
            return "You Selected ",inputGesture," and computer selected ",ComputerGesture,"So You Win "
        elif inputGesture == "rock":
            return "You Selected ",inputGesture," and computer selected ",ComputerGesture,"So You Lose "
        
    elif ComputerGesture == "scissor":
        
        if inputGesture == "scissor":
            return "Both Selected ",ComputerGesture," Its A Draw"
        elif inputGesture == "rock":
            return "You Selected ",inputGesture," and computer selected ",ComputerGesture,"So You Win "
        elif inputGesture == "paper":
            return "You Selected ",inputGesture," and computer selected ",ComputerGesture,"So You Lose "
        
    else:
        return("Invalid Input")
        
print(gameOutPut("paper"))