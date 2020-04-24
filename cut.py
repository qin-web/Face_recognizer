s = 'wjc123'
name = ''.join([i for i in s if not i.isdigit()])
print(name)