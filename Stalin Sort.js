// Expirements - Stalin sort

var arr = [
    4, 15, 23, 42, 8, 16, 38, 29, 11, 50,
    3, 27, 19, 32, 12, 41, 37, 6, 21, 34
]

function stalinSort(array) {
    var nArray = array
    var last
    var operatable = 0
    var it = 0
    var finished = false
    var itsafchan = 0
    while (!finished) {
        var curVal = nArray[operatable]
        if (last && (curVal < last)) {
            nArray.splice(operatable, 1)
            nArray.unshift(curVal)
            last = undefined
            itsafchan = 0
        } else {
            if (operatable < nArray.length) {
                operatable++;
            } else {
                operatable = 0
            }
            last = curVal
        }
        itsafchan++
        it++
        if (itsafchan > array.length) {
            finished = true
        }
    }
    return [nArray, it]
}

console.log(stalinSort(arr))
