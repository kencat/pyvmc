#nucleus={H:{1,1,0,0,0}}
#electrons={e1:{1,1,1}}
#r=nucleus[H][2]
#python3
import numpy as np

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

def V(r):  #potential
  return -(e**2)/(4*pi*epsilon_0*r)

H=nucleu()
e_1=electron()

r=((H.x-e_1.x)**2+(H.y-e_1.y)**2+(H.z-e_1.z)**2)**0.5
print("r:",r,"V:",V(r))

minx=-5
maxx=5
num=1000
step=(maxx-minx)/num
x = np.linspace(minx,maxx,num)
r = abs(x)

class phi: #orbitals
  def __init__(self,r):
    self.r=r
    self.a=1.0 #Bohr radius
    self.A=(1/((self.a)**3)*pi)**(0.5)
    self.psi=self.A*np.exp(-(r/self.a))


e_STO=phi(r).psi.sum()/num
print("e_STO:", e_STO)
