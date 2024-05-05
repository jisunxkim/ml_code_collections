def zero_sum(numbers):
    res =[]
    subset= []

    def backtrack(idx):
        if (len(subset) != 0) and sum(subset) == 0:
            res.append(subset.copy())
        if idx >= len(numbers):
            return

        subset.append(numbers[idx])
        backtrack(idx+1)
        subset.pop()
        backtrack(idx+1)

    backtrack(0)

    return res

numbers = [1,-2,6,7,1]
print(zero_sum(numbers))

