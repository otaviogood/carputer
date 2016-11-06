
throttle_max_range_map = [
    [80,0], # if t <= 80 -> o=0 # Breaking:
    [82,1], # elif t <= 82 -> o=1
    [84,2], # elif t <= 84 -> o=2
    [86,3], # elif t <= 86 -> o=3
    [87,4], # elif t <= 87 -> o=4 # Breaking ^
    
    [96,5], # elif t <= 96 -> o=5 # Neutral

    [97,6], # elif t <= 97 -> o=6 # Forward:
    [98,7], # elif t <= 98 -> o=7
    [99,8], # elif t <= 99 -> o=8
    [100,9], # elif t <= 100 -> o=9
    
    [101,10], # elif t <= 101 -> o=10
    [102,11], # elif t <= 102 -> o=11
    [105,12], # elif t <= 105 -> o=12
    [107,13], # elif t <= 107 -> o=13
    [110,14]  # elif t <= 110 -> o=14
]

map_back = {5:90}

def to_throttle_buckets(t):
    t = int(float(t)+0.5) #nearest

    for max_in_bucket,bucket in throttle_max_range_map:
        if t <= max_in_bucket:
            return bucket
    return 14

def from_throttle_buckets(t):
    t = int(float(t)+0.5)
    for ibucket,(max_in_bucket,bucket) in enumerate(throttle_max_range_map):
        if t == bucket:
            if map_back.has_key(bucket):
                return map_back[bucket]

            return max_in_bucket

    return 100 # Never happens, defensively select a mild acceleration

if __name__ == '__main__':

    for i in range(120):
        print 'Throttle=',i, '-> Bucket=', to_throttle_buckets(i)

    print 'Inverse mapping'
    for i in range(15):
        print 'Bucket=',i, '-> Throttle=', from_throttle_buckets(i)
