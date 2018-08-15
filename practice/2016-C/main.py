#!/usr/bin/env python
# -*- coding: utf-8 -*-

from modules.Case import Case


def main():
    case = Case()
    case.load_case("samples/learning/case001_input.txt")
    case.init_argvs()
    print(case.first())

if __name__ == "__main__":
    main()
