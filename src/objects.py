#nucleus={H:{1,1,0,0,0}}
#electrons={e1:{1,1,1}}
#r=nucleus[H][2]
#python3

e=1.6e-19                        #C
epsilon_0=8.854187812813e-12 #F*(m**−1)
pi=3.1415926 

class nucleu:
  def __init__(self):
    self.name ="H"
    self.charge = 1
    self.weight = 1
    self.x = 0
    self.y = 0
    self.z = 0
  def speak(self):
    print("I am  %s "%self.name)

class electron:
  def __init__(self):
    self.x = 1
    self.y = 1
    self.z = 1
    self.spin = 1 #1 is up 0 is down

