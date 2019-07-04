import random
import objects

r=1

def single_wavefunction(n,e):
  p=random.uniform(0,1)
  e.x=p*r+n.x
  e.y=p*r+n.y
  e.z=p*r+n.z


