# Code Guidelines

We mostly follow the google python style guide.

https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings

Notable intended exceptions are:
 
 * use \`my_variable\` to refer to a variable in comments. This helps distinguish words that refer to variables from 
 normal words of the sentence. 
 * use numpy style argument description, which uses this form:
```
def foo(bar):
    """Do something cool
    
    Args:
        bar: string or sequence of strings
            Crucial input.
    """
``` 

Furthmore:
 *  Note that not all souce code is yet styled in this way.


