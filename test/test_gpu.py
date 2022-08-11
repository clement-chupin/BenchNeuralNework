import torch

print("Nombre de gpu : {}".format(torch.cuda.device_count()))
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))

