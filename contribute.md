# Code Guidelines

We mostly follow the google python style guide.

https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings

Notable and intended exceptions are:
 
 * use \`my_obj\` to refer to a variable, class or function in comments, i.e. use backticks This helps distinguish words that refer to variables from 
 normal words of the sentence. 
 * use numpy style argument description, which uses this form:
```
def foo(bar):
    """Do something cool
    
    Args:
        bar: string or sequence of strings
            semantic description of bar
    """
``` 

Furthmore:
 *  Note that not all souce code is yet styled in this way... it's work in progress


