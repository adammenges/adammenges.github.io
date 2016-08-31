---
layout:     post
title:      Software Engineering Interviews
date:       2013-03-17
summary:    A little while ago I was graduating college, and looking for a job. During that time, I spent a lot of my days gathering knowledge that could help me during technical interviews. I wanted to be able to pull off the perfect one, and come off as professional, knowledgeable, and above all, kickass.
permalink: software-engineering-interviews
---

A little while ago I was graduating college, and looking for a job. During that time, I spent a lot of my days gathering knowledge that could help me during technical interviews. I wanted to be able to pull off the perfect one, and come off as professional, knowledgeable, and above all, kickass.

Alright, let's jump right in and look at the easy gotcha questions. These are simple short-answer questions an interviewer can ask that will help them quickly gain some knowledge about you.

* What's the difference between GET and POST?
* Explain the REST architecture. 
* Why would someone need to overwrite the .equals() method in a Java class?
* Explain everything that happens when type in a url into your web browser.
* What is the main difference between HTTP 1.0 and 1.1?
* What is polymorphism?
* Implement a swap function without using a temp variable.
* What is the difference between an interface and an abstract method?
* How is a Hash Table typically implemented?
* Describe a binary tree structure and it's variants.
* What is Late Static binding?
* How is the each method implemented in Ruby? How would you do it?
<br/>

Next, here are some of the more involved programming related questions you may get:

## Question 1

One of the most basic and common questions is "Show me an method that checks to see if a string has any duplicates."

``` java
public boolean inUnique(string str) {
     boolean[] chars = new boolean[256]; // Size only works with ASCII, increase otherwise.
     for(int i=0; i<str.size(); i++){
          if(chars[str.charAt(i)]) return false;
          chars[str.charAt(i)] = true;
     }
     return true;
}
```

This approach is simple, and requires O(n) time. If space is more important, testing every char against every other char is the way to go:

``` java
public boolean isUnique(string str) {
     for(int i=0; i<str.size(); i++){
          for(int j=0; j<str.size(); j++) {
               if(i==j) continue;
               else if(str.charAt(j) == str.charAt(i)) return false;
          }
     }
     return true;
}
```

And yet another way, using a hash:

``` python
from collections import defaultdict
 
def isUnique(string):
  seen = defaultdict(lambda: False)
  for x in string:
    if not seen[x]: seen[x] = True
    else: return False
  return True
```

You could also use a bit vector to use even less space.

## Question Two

Another great one is the FizzBuzz problem, popularized by Jeff Atwood in [this blog post](http://www.codinghorror.com/blog/2007/02/why-cant-programmers-program.html).

Basically the idea is to print every number from 1 to n, however, print 'fizz' every time the number is divisible by 3, to print 'buzz' every time the number is divisible by 5, and the print 'fizzbuzz' every time the number is divisible by both 3 and 5 (15). These numbers could change to two different prime numbers. This doesn't change how you would solve the problem.

One of my favorite solutions is:

``` python
def fizzbuzz(n):
    print "\n".join([('Fizz'*(not i%3) + 'Buzz'*(not i%5)) if ((not i%3) or (not i%5)) else str(i) for i in xrange(1, n+1)])
```

Another great one is:

``` python
def fizzbuzz(n):
    for i in xrange(1, n+1): print [i, 'Fizz', 'Buzz', 'FizzBuzz'][(i%3 == 0) + 2 * (i % 5 == 0)]
```

Something else I've seen is building on this for a FizzBuzzBazz problem. This just adds the phrase "Bazz", linked to the next prime, say 7. Everything else stays the same. Usually this is used to show why some solutions may be incorrect, or how they can be simplified.

## Question Three
Another common one is to find the nth [fibonacci number](http://en.wikipedia.org/wiki/Fibonacci_number).

``` python
def fib(n):
  return reduce(lambda x, y: [x[1], x[0] + x[1]], xrange(n-2), [0, 1])[1]
```

## Question Four

Next, the parentheses problem. In this one, the basic idea is to create a method that takes a string as input, and returns a boolean. True if the parentheses in the string are valid, false if not. E.g. () is valid. )( is not. There are lots of solutions to this problem. One thing to keep in mind: a stack lends itself perfectly to this problem. Here's a solution using a one:

``` ruby
def validParentheses? str
  a = []
  str.each_char do |x|
    if x == '('
      a.push x
    elsif x == ')'
      return false if a.pop == nil 
    end
  end
  a.empty?
end
```

Alternate solotion, given to me by [@Sirupsen](https://twitter.com/Sirupsen):

``` ruby
def balanced?(string)
  string.each_char.inject(0) { |open, char|
    return false if open < 0
    char == '(' ? open + 1 : open - 1
  } == 0
end
```

It'd be pretty easy to extend this to work with many kinds of brackets. Something like this would do:

``` python
def balancedParens(s):
  stack, opens, closes = [], ['(', '[', '{'], [')', ']', '}']
  for c in s:
    if c in opens:
      stack.append(c)
    elif c in closes:
      try:
        if opens.index(stack.pop()) != closes.index(c):
          return False
      except (ValueError, IndexError):
        return False
  return not stack
```

Also, [here's a similar problem](http://www.boi2012.lv/data/day1/eng/brackets.pdf).

## Question Five

Alright, next, another common one I've seen goes something like 'find all possible letter representations for a given phone number.'

``` python
num_map = {'0': '0', '1': '1', '2': 'ABC', '3': 'DEF', '4': 'GHI', '5': 'JKL', '6': 'MNO', '7': 'PQRS', '8': 'TUV', '9': 'WXYZ'}
 
def get_number_representations(n):
    return itertools.product(*[num_map[x] for x in n])
```

## Question Six

What are the different types of joins? Please explain how they differ and why certain types are better in certain situations. It's important to know these things.

[There](http://www.codinghorror.com/blog/2007/10/a-visual-explanation-of-sql-joins.html). [Are](http://msdn.microsoft.com/en-us/library/zt8wzxy4.aspx). [Bunches](http://en.wikipedia.org/wiki/Join_(SQL)). [Of](http://www.w3schools.com/sql/sql_join.asp). [Places](http://stackoverflow.com/a/17946222/458961). [Online](http://stackoverflow.com/questions/6294778/mysql-quick-breakdown-of-the-types-of-joins). [That](http://www.made2mentor.com/2011/02/different-types-of-joins/). [Show](http://www.datamartist.com/sql-inner-join-left-outer-join-full-outer-join-examples-with-syntax-for-sql-server). [What](http://beginner-sql-tutorial.com/sql-joins.htm). [Joins](http://dwhlaureate.blogspot.com/2012/08/joins-in-oracle.html). [Are](http://www.codeproject.com/Articles/33052/Visual-Representation-of-SQL-Joins).

## Question Seven

Explain the following terms: virtual memory, page fault, thrashing.

Virtual memory is a computer system technique which gives an application program the impression that it has contiguous working memory (an address space), while in fact it may be physically fragmented and may even overflow on to disk storage. Systems that use this technique make programming of large applications easier and use real physical memory (e.g. RAM) more efficiently than those without virtual memory. - [Wikipedia](http://en.wikipedia.org/wiki/Virtual_memory)

Page Fault: A page is a fixed-length block of memory that is used as a unit of transfer between physical memory and external storage like a disk, and a page fault is an interrupt (or exception) to the software raised by the hardware, when a program accesses a page that is mapped in address space, but not loaded in physical memory. - [Wikipedia](http://en.wikipedia.org/wiki/Page_fault)

Thrash is the term used to describe a degenerate situation on a computer where increasing resources are used to do a decreasing amount of work. In this situation the system is said to be thrashing. Usually it refers to two or more processes accessing a shared resource repeatedly such that serious system performance degradation occurs because the system is spending a disproportionate amount of time just accessing the shared resource. Resource access time may generally be considered as wasted, since it does not contribute to the advancement of any process. In modern computers, thrashing may occur in the paging system (if there is not ‘sufficient’ physical memory or the disk access time is overly long), or in the communications system (especially in conflicts over internal bus access), etc. - [Wikipedia](http://en.wikipedia.org/wiki/Thrash_(computer_science))


## Question Eight
Write a method to shuffle a deck of cards. It must be a perfect shuffle - in other words, each 52! permutations of the deck has to be equally likely. You can assume you have a true random number generator.

``` java
public void shuffleCards (int[] cards){ 
  int temp, index;
  for (int i = 0; i < cards.length; i++){
    index = (int) (Math.random() * (cards.length - i)) + i;
    temp = cards[i];
    cards[i] = cards[index];
    cards[index] = temp;
  }
}
```

## Question Nine
Finally, for this section, there's the good old 'is binary search tree' question. Simple enough solution:

``` python
def isBST(node, minVal, maxVal):
    if node is None: return True
    if not minVal <= node.val <= maxVal: return False
    return isBST(node.left, minVal, node.val) and isBST(node.right, node.val, maxVal)
```

## Question Ten
The bottle one. You have a five quart jug and a three quart jug, an unlimited supply of water, and nothing else. How would you come up with ***exactly*** four quarts of water?

Like this:

* Fill 5
* Pour into 3
* Pour out 3
* Pour 2 from 5 into 3
* Fill 5
* Pour out 1 from 5 into 3
<br/>

Or like this:

* Fill 3
* Pour into five
* Fill 3
* Pour 2 into 5 from 3
* Pour out 5
* Pour 1 into 5
* Fill 3
* Pour 3 into 5

Answering this may also just prove you've watched Die Hard. 😏

<div style="position: relative; padding-bottom: 56.25%; padding-top: 35px; height: 0; overflow: hidden;"><center><iframe width="420" height="315" style="position: absolute; top:0; left: 0; width: 100%; height: 100%;" src="//www.youtube.com/embed/lZ64IR2bz5o" frameborder="0" allowfullscreen></iframe></center></div>

<br />

## Question Eleven
A bunch of folks are on an island. A magical genie comes down and gathers everyone together and places a magical hat on at least one person's head. The hat can be seen by other people, but not by the person wearing it. To remove the hat, those (and only those who have a hat) must dunk themselves underwater at exactly midnight. If there are x people and y hats, how long does it take the folks to remove the hats? These people cannot talk to each other.

Answer:

y-1 nights. Simple, no? This is because if there's one guy wearing a hat, he will see no one around him wearing a hat, it must be him, so he'll go dunk his head in the water at midnight. Then, if two people are wearing hats, each will see someone else wearing a hat, wait a night, then see that the other didn't go dunk their head, so it must be them and the other person with a hat on. And so on and so forth. 


## Question Twelve
Here's one I heard in college, from one of my favorite professors: There is a building of 100 floors. Eggs have a certain strength to them, and will break past floor x. Find the minimal n, with two eggs. By the end you can break both eggs.

Answer:

The most basic attempt is simple, drop the first egg from sqrt(floors), then increment. So 10, then 20, then 30, etc. Then once it breaks at floor x, drop it from floor (x*10)-9, then (x*10)-8, and so on. This puts the worst case minimum at 19.

In that case, we dropped each egg the same number of times no matter what floor the first broke at, a better way to do it is try and get each try of the first egg to reduce the amount you have to drop the second egg, resulting in a true minimum. This means solving for: (x) + (x-1) + (x-2) + (x-3) ... + 1 = 100 --> x=14. So we'd drop the first egg at 14, then 27, 39, etc. Our worst case minimum is 14.

Also, [dynamic programming](http://en.wikipedia.org/wiki/Dynamic_programming#Egg_dropping_puzzle).

---------------------------

Now, I'm not saying these are the best types of questions to be asking candidates (they're not), but they are the types of questions I get in interviews. For whatever reason. Doesn't really matter, it's your job right now to ace whatever they throw at you, and at least for just-out-of-college software engineering jobs, these tend to be the questions thrown at candidates.

The propose of this post is to get your mind going in the right direction; what you need to know to ace an interview, or at least the technical part. This is not meant to be the only document you read. From here another good next step could be to look at previous [Google Code Jams](https://code.google.com/codejam/), [Top Coder](http://topcoder.com), or (my favorite) [Project Euler](https://projecteuler.net/problems). Another interviewing technique I've seen is where the interviewer has already written the spec file, and it's your job to make everything pass. Make sure to know you TDD/BDD stuff.

Another topic outside the scope of this post is how to act, this may have an even greater effect on your chances than having your technical wits. I wouldn't have gotten my current job going in there wearing a suit, I came in wearing a sweatshirt and jeans, immediately fitting in with everyone else.

Last, but definitely not least, have fun! Interviews can be fun if you let them. Don't stress, enjoy meeting some new people and talking tech. Happy interviewing!
