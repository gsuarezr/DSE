import subprocess 
 
for i in range(2,6):
# Run script1.py 
    subprocess.run(["python", "ising.py",f"{i}",f"5","1"]) 
