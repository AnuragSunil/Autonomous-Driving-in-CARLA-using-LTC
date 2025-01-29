import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
maps = client.get_available_maps()
print("Available maps:", maps)