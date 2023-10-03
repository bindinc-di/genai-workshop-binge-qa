## %
import random, string

for _ in range(10):
    randdom_string = "".join(random.choices(string.ascii_letters + string.digits, k=32))
    print(randdom_string)