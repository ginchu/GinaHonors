import os

os.system("ls carbon_energies_forces/monolayer > m_ls.txt")

m = open("m_ls.txt", "r")
filenames = m.read().split()

f = open("target_e.txt", "a+")
g = open("quip_e.txt", "a+")
f.close()
g.close()

t = open("filenames.txt", "a+")

n = 25
idx = 6
for fn in range(0, len(filenames), idx):
    xyz = "carbon_energies_forces/monolayer/" + filenames[fn]
    t.write(xyz + "\n")
    os.system("grep 'Energy=' " + xyz + " | sed -n -e 's/^.*Energy=//p' >> target_e.txt")
    os.system("quip atoms_filename=" + xyz + " param_filename=../Carbon_GAP_20/Carbon_GAP_20_potential/Carbon_GAP_20.xml E | grep -m1 'Energy=' | sed -n -e 's/^.*Energy=//p' >> quip_e.txt")
    if (n-1)*idx == fn:
        break
t.close()

