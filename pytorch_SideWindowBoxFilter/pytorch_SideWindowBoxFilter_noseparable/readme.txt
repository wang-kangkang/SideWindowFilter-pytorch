(1)run pytorch_SideWindowBoxFilter
(2)run main.m
(3)see figure, check diff. the diff comes from python float predision. for example:matlab:min(abs(0.0123),abs(-0.0124)),  python:min(abs(0.0124),abs(-0.0123))