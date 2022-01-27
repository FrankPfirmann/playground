class Logger:
    def __init__(self, name):
        self.name = name

    def write(self, *data):
        data_str = ""
        for point in data:
            data_str += str(point) + "\t"
        data_str += "\n"

        with open("data/"+self.name, "a") as myfile:
            myfile.write(data_str)
