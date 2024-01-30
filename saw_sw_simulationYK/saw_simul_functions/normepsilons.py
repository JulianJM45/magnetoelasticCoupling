
eps = {}

eps['xx'] =  (68.5599918997257+17.122278103400706j)
eps['xy'] =  (7.518160375585753-30.322126695128436j)
eps['xz'] =  (17.754438781091295-84.71784906405115j)
eps['yz'] =  (12.629232043550473+4.6202909648719j)



max_abs_real = max(abs(value.real) for value in eps.values())
max_abs_imag = max(abs(value.imag) for value in eps.values())
max_abs = max(max_abs_real, max_abs_imag)
eps = {key: (value.real / max_abs) + (value.imag / max_abs) * 1j for key, value in eps.items()}



eps['xx'] = round(eps['xx'].real, 2) + round(eps['xx'].imag, 2) * 1j
eps['xy'] = round(eps['xy'].real, 2) + round(eps['xy'].imag, 2) * 1j
eps['xz'] = round(eps['xz'].real, 2) + round(eps['xz'].imag, 2) * 1j
eps['yz'] = round(eps['yz'].real, 2) + round(eps['yz'].imag, 2) * 1j



print("eps['xx'] = ", eps['xx'])
print("eps['xy'] = ", eps['xy'])
print("eps['xz'] = ", eps['xz'])
print("eps['yz'] = ", eps['yz'])