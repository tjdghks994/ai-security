# 기본 출력
print("Hello AI world!")                        # () 안의 내용 출력

# 사칙연산
a = 2
b = 3
c = 4
d = 9

print("\n")
print(a+b, a-b, a*b, a/c, d%a, b**c, d//a)      # %: 나머지, //: 몫, **: 제곱

# list = arrary
print("\n")
list_a = [0,1,2]                                # list에 0,1,2 추가  
print(list_a)

list_a.append(3)                                # list의 마지막에 3 추가
print(list_a)

list_a.append("char")                           # list의 마지막에 char 추가, 문자열 추가 가능
print(list_a)

print(list_a[:-2])                              # list의 끝에서 2번째까지는 제외하고 출력

del list_a[-1]                                  # list의 마지막 항목 제거
print(list_a)

list_a.pop()                                    # list의 마지막 항목 제거, del과 유사한 기능
print(list_a)

# dictionary
print("\n")
dict_a = {                                      # dictionary 생성, 왼쪽을 key, 오른쪽을 value라고 부름 
    "ai" : 0,
    "sec" : 1,
    "sh" : 2,
    "yr" : 3,
    "jw" : 4,
    "key" : "vaule"
}
print(dict_a)
print(dict_a["sh"])                             # dic에서 sh의 value 출력
# print(dict_a[2])                              # value 형태로 출력은 불가능!
del dict_a["key"]                               # "key"라는 이름을 가진 key 제거
print(dict_a)

dict_a["class"] = 5                             # dic 마지막에 "class"란 키에 5라는 value를 갖는 항목 추가
print(dict_a)


# for문
print("\n")
sum = 0
for i in range(10):                             # i는 0부터 시작, 1씩 늘어나며  10까지 반복
    sum += i                                    # sum 변수에 i를 더함
    print(sum)

print("\n")
for j in list_a:                                # lsit_a를 j(= 0)번째부터 출력
    print(list_a[j]) 

# while문
print("\n")
x = 0
while(x < 10):                                  # () 안의 조건을 만족할 때까지 반복
    x += 1
    print(x)

# boolen function
print("\n")
for i in dict_a:
    if i == "sh":                               # 해당 조건을 만족하면 아래 코드 실행
        print("yes I'm sunghwan")
    elif i == "yr":                             # else if문, if문 안에서 두 개 이상의 조건이 필요할 때 사용
        print("yes I'm yeryoung")
    else:                                       # 위의 모든 조건에 만족하지 않을 경우 아래 코드 실행
        print("??")

# 함수 만들어 쓰기
print("\n")
def factorial(n):                               # factorial 함수 실행
    result = 1                                  # 0부터 시작하면 모두 0이 나오므로 1로 초기화
    for i in range(1, n+1):                     # range 안의 왼쪽 값부터 오른쪽 값까지 실행
        result = result * i
    return result

print(factorial(6))

# class와 object
print("\n")
class sunghwan(object):                         # sunghwan 이란 class 생성, class 안에는 다양한 함수를 포함 시킬 수 있음
    def __init__(self, name, age):
        self.name = name
        self.age = age
        self.time = 0

    def call(self):
        self.time += 1

TA = sunghwan("sunghwan", 26)
TA.call()

print(TA.name)
print(TA.age)
print(TA.time)

TA.call()
print(TA.time)

# https://github.com/GunhoChoi/PyTorch-FastCampus/blob/master/01_DL%26Pytorch/0_Python_Tutorial.ipynb
# 해당 github을 참조하여 구성