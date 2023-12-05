def median(numbers):
    numbers = sorted(numbers)
    if len(numbers) % 2 == 0:
        middle2 = len(numbers) // 2
        middle1 = middle2-1
        res = (numbers[middle1] + numbers[middle2]) / 2
    else:
        middle = len(numbers)//2
        res = numbers[middle]
    return res
