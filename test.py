shape = str(input())
size = int(input())

if shape == "a":
    def printPatternX(size):
        y = 1
        for j in range(size):
            for i in range(size):
                if (y == 1 or y == size):
                    print("#", end="")
                if ((y != 1 and y != size) and i == 0):
                    print("#" + (" " * (size - 2)) + "#")
            if y == 1:
                print("")
            y += 1

if shape == "b":
    def printPatternX(size):
        y = 1
        for j in range(size):
            for i in range(size):
                if (y == 1 or y == size):
                    print("#", end="")
                if ((y != 1 and y != size) and i == y):
                    print(" " * (y - 1) + "#")
            if y == 1:
                print("")
            y += 1

if shape == "c":
    def printPatternX(size):
        y = 1
        r = size
        for j in range(size):
            for i in range(size):
                if (y == 1 or y == size):
                    print("#", end="")
                if ((y != 1 and y != size) and i == (r - 1)):
                    print(" " * (r - 1) + "#")
            if y == 1:
                print("")
            y += 1
            r -= 1

if shape == "d":
    def printPatternX(size):
        y = 1
        print("#" * size)
        for j in range(size - 2):
            for i in range(size):
                if (i == y or i == (size - y - 1)):
                    print("#", end="")
                else:
                    print(" ", end="")
            print("")
            y += 1
        print("#" * size)

printPatternX(size)
