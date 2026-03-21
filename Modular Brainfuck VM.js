let defaultlog
let defaultinput = function () { return "a"; };

var vm = {
    "brainfuck": function (code, log, input) {
        if (log === void 0) { log = defaultlog }
        if (input === void 0) { input = defaultinput }
        var mem = [0];
        var pointer = 0;
        var pc = 0;
        var jumpTable = {};
        var stack = [];
        for (var i = 0; i < code.length; i++) {
            if (code[i] === '[')
                stack.push(i);
            else if (code[i] === ']') {
                if (stack.length === 0)
                    throw new Error("Unmatched ] at ".concat(i));
                var start = stack.pop();
                jumpTable[start] = i;
                jumpTable[i] = start;
            }
        }
        if (stack.length > 0)
            throw new Error("Unmatched [ at ".concat(stack.pop()));
        while (pc < code.length) {
            var ins = code[pc];
            switch (ins) {
                case "+":
                    mem[pointer]++;
                    if (mem[pointer] > 255) {
                        mem[pointer] = 0;
                    }
                    break;
                case "-":
                    mem[pointer]--;
                    if (mem[pointer] < 0) {
                        mem[pointer] = 255;
                    }
                    break;
                case ">":
                    pointer++;
                    if (pointer >= mem.length) {
                        mem.push(0);
                    }
                    break;
                case "<":
                    pointer--;
                    if (pointer < 0) {
                        pointer = mem.length - 1;
                    }
                    break;
                case "[":
                    if (mem[pointer] === 0)
                        pc = jumpTable[pc];
                    break;
                case "]":
                    if (mem[pointer] !== 0)
                        pc = jumpTable[pc];
                    break;
                case ".":
                    log(String.fromCharCode(mem[pointer]));
                    break;
                case ",":
                    var char = input();
                    mem[pointer] = char.charCodeAt(0) || 0;
                    break;
            }
            pc++;
        }
        return mem;
    },
    "ook": function (code, log, input) {
        var _a;
        vm.brainfuck(((_a = code.match(/Ook[.?!] Ook[.?!]/g)) === null || _a === void 0 ? void 0 : _a.map(function (p) { return ({ 'Ook. Ook?': '>', 'Ook? Ook.': '<', 'Ook. Ook.': '+', 'Ook! Ook!': '-', 'Ook! Ook.': '.', 'Ook. Ook!': ',', 'Ook! Ook?': '[', 'Ook? Ook!': ']' })[p]; }).join('')) || '', log, input);
    }
};
if (typeof module != "undefined" && module.exports) {
    module.exports = vm;
    defaultlog = process.stdout.write
} else {
    let laststring = ""
    defaultlog = (str) => {
        if (str == "\n") {
            console.log(laststring)
            laststring = ""
        } else {
            laststring += str
        }
    }
    defaultinput = prompt
}
