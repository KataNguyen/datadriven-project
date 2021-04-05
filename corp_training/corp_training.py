###############################################################################

# while

# tạo một list chứa các phần tử từ 1 đến 100
integers = []
x = 1
while x <= 100:
    integers += [x]
    x += 1

# tạo một list chứa 20 số hạng đầu tiên của dãy Fibonacci
a,b = 0,1
fibo = []
while len(fibo) < 20:
    a,b = b,a+b
    fibo += [a]

# tạo một list chứa 20 số chính phương đầu tiên
integers = []
x = 1
while x <= 20:
    integers += [x]
    x += 1

i = 0
squared_list = []
while len(squared_list) < 20:
    squared_num = integers[i] ** 2
    squared_list += [squared_num]
    i += 1

# Cho n=1, cần phải chia n cho 2 tối thiểu bao nhiêu lần để kết quả thu được
# nhỏ hơn 1/1000
n = 1
count = 0
print(n)
while n >= 1/10000:
    n /= 2
    count += 1
    print(n)

# Một trái phiếu trả lãi 8% một năm, cần nắm giữ bao lâu để nhân 3 tài sản
asset = 100
years = 0
while asset < 300:
    asset *= (1+0.08)
    years += 1

###############################################################################

# for

# Dự đoán kết quả của đoạn code sau:
words = ['car', 'window', 'pencil']
for w in words:
    print('This is a ' + w)

# range function
list(range(0, 20))
list(range(0, 20, 2))
list(range(0, 20, 3))
list(range(20, 0, -1))
list(range(20, 5, -2))
list(range(0, 20, -1))

# Tạo một list các phần tử từ 1 đến 100

my_list = list(range(1,101,1))
print(my_list)

my_list = []
for n in range(1,101,1):
    my_list += [n]
print(my_list)

# Tạo một list các số chẵn từ -50 đến 50
my_list = list(range(-51,51,2))
print(my_list)

my_list = []
for n in range(-51,51,2):
    my_list += [n]

# Tạo một list chứa 20 số chính phương đầu tiên
my_list = []
for num in range(1,21):
    my_list += [num**2]

# Bài 7:
flattened_list = []
nest_list = [[1,4,9,16,25],
             ['a','b','c','d','e'],
             ['X1','X2','Y1','Y2','Y3']]
for sub_list in nest_list:
    for n in sub_list:
        flattened_list += [n]
print(flattened_list)

# Tung một con xúc xắc 3 lần, liệt kê tất cả các trường hợp
i = 1
for k in range(1,7):
    for l in range(1,7):
        for m in range(1,7):
            print('========================\n')
            print('Lần thử thứ ' + str(i) +':')
            print('----')
            print('Số điểm của lần tung 1: ' + str(k))
            print('Số điểm của lần tung 2: ' + str(l))
            print('Số điểm của lần tung 3: ' + str(m))
            i += 1

# Tung một con xúc xắc 3 lần, tính xác suất để tổng số điểm sau 3 lần tung
# nhỏ hơn 10
total_trials = 0
passed_trials = 0
for k in range(1,7):
    for l in range(1,7):
        for m in range(1,7):
            total_trials += 1
            total_score = k + l + m
            if total_score < 10:
                passed_trials += 1
print('Xác suất cần tìm là: ' + str(passed_trials/total_trials*100) + '%')

# Bài 10:

# Bước 1: Tạo một bộ bài
card_deck = list(range(1,14)) * 4
for i in range(len(card_deck)):
    if card_deck[i] > 10:
        card_deck[i] = 10

# Bước 2: Chơi tất cả các game có thể có
points = []
for card1 in card_deck:
    remaining = card_deck.copy()
    remaining.remove(card1)
    for card2 in remaining:
        remaining.remove(card2)
        for card3 in remaining:
            raw_point = card1 + card2 + card3
            point = raw_point % 10
            points += [point]

# Câu a:
passed_trials = 0
for point in points:
    if point == 9:
        passed_trials += 1
print('Xác suất cần tìm là: ' + str(passed_trials/len(points)*100) + '%')

# Câu b:

x = 15
if x < 10:
    print(x)

if x % 2 == 0:
    print('x is even')
elif x > 20:
    print('x is greater than 20')
else:
    print('x is odd and less than 20')


if x % 2 == 0:
    print('x is even')
if x > 20:
    print('x is greater than 20')
else:
    print('x is odd and less than 20')

x = 100
my_list = []
while x >= 0:
    my_list += [x]
    x -= 1
    if x == -1:
        continue
