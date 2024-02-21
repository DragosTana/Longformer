



class Parent():
    def __init__(self):
        print("Parent constructor")

    def parent_method(self):
        print("Parent method")
        
        
class Child(Parent):
    def __init__(self):
        print("Child constructor")
        super().__init__()

    def parent_method(self):
        print("Child method")
        super(Child, self).parent_method()
        
c = Child()
c.parent_method()