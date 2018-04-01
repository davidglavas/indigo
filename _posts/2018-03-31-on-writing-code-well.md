---
title: "On Writing (Code) Well"
layout: post
date: 2018-03-31 15:22
mathjax: true
headerImage: true
tag:
- code complete 2
- on writing well
- software engineering advice
star: true
category: blog
author: davidglavas
description: I point out similarities between programming and writing nonfiction based on interchangeable advice in Steve McConnell’s Code Complete and William Zinsser’s On Writing Well.
---

<p align="center">
  <img src="https://raw.githubusercontent.com/davidglavas/davidglavas.github.io/master/_posts/Figures/2018-03-31-on-writing-code-well/FrontCovers.jpg">
</p>


## TL;DR
I point out similarities between programming and writing nonfiction based on interchangeable advice in Steve McConnell’s *Code Complete* and William Zinsser’s *On Writing Well*. For the identified similarities—clarity of thought, simplicity, and the importance of iterations—I elaborate on McConnell’s advice for writing code well.

The goal of this post is to share low-hanging fruits, that is, practical and immediately applicable advice any programmer can benefit from. I read *On Writing Well* and *Code Complete* in parallel which taught me some similarities. To ~~justify the time I spent procrastinating~~ keep this post interesting, I relate software construction to nonfiction writing and use the relationship as a basis for McConnell’s advice. First I’ll briefly introduce the books and give you a reason to believe the relationship exists. Then, I’ll go over the three main similarities—clarity of thought, simplicity, and the importance of iterations—that I found to be especially relevant for constructing software.

Let’s take a brief look at the two books.

## The Books

Zinsser’s *On Writing Well* is all about expressing oneself with clarity, simplicity, brevity and humanity. It gives a glimpse into the habits of a professional writer and covers general advice such as perseverance, consistency, how to write a good leads and endings, how to not sound emotionless or like a copycat, and much more. He then shows how to apply this advice to various forms such as interviews, travel articles, memoirs, science, technology, business writing, humor and more.

McConnell’s *Code Complete* is a guided tour on lots of widely used development practices. It covers all kinds of issues related to software construction—from variables and statements to code tuning and collaborative construction. Besides learning a ton of new things, I enjoyed seeing tricks that I was using for some time now—especially those which I did unconsciously and never bothered to stop and think more about.
To all those little things—like minimizing the distance in lines between the initialization of variables and their references—McConnell gives names such as live time and span. He manages to give names to things that most of us do intuitively but don’t consciously think about. Regardless of the specific names, the descriptions allow the reader to put a finger on what he already knows while picking up lots of new tricks along the way.

So is there a connection between programming and nonfiction writing? If so, then we should be able to find parts in the books with the same underlying ideas. 

For the following four quotes, try and guess to which of the two books each of them belongs:

> Look for the clutter and prune it ruthlessly. Be grateful for everything you can throw away. Are you hanging on to something useless just because you think it’s beautiful? Simplify, simplify.

> The point is that you have to strip your work down before you can build it back up. You must know what the essential tools are and what job they were designed to do.

> Sometimes you will despair of finding the right solution—or any solution. You’ll think, “If I live to be ninety I’ll never get out of this mess.” I’ve often thought it myself. But when I finally do solve the problem it’s because I’m like a surgeon removing his 500th appendix; I’ve been there before.

> When you find yourself in such a situation, look at the troublesome element and ask, “Do I need it at all?” Probably you don’t. It was trying to do an unnecessary job all along.

If you guessed *On Writing Well* four times then well done. The point is that these statements could easily fit into both books. Let’s interpret the interchangeability of these (and many other) statements as proof for the existence of a connection between programming and non-fiction writing. 

In the rest of this post I’ll cover McConnell’s advice on three points which are mentioned repeatedly throughout both books—clarity of thought, simplicity, and the importance of iterations.

## 1. Clarify your thoughts first


<p align="center">
  <img src="https://i1.wp.com/www.susankingsleysmith.com/wp-content/uploads/2015/08/image1.jpg?w=750">
</p>

Clear minds tend to write clear sentences and produce clear code.

> Writers must therefore constantly ask: what am I trying to say? Surprisingly often they don’t know. Then they must look at what they have written and ask: have I said it? Is it clear to someone encountering the subject for the first time? If it’s not, some fuzz has worked its way into the machinery. The clear writer is someone clearheaded enough to see this stuff for what it is: fuzz.” 
>
> — <cite>William Zinsser</cite>  
<br/>

More complicated structures require more careful planning, they also benefit from different levels of planning. McConnell says that “from a technical point of view, planning means understanding what you want to build so that you don’t waste money building the wrong thing.“ Investing time into precisely documenting requirements in order to avoid building the wrong features and therefore satisfying the wrong requirements is a form of planning. The same goes for system, object and any other kind of design. 

In a sense, planning is a form of clarifying our thoughts. We don’t talk about requirements and create time consuming design documents for their own sake. We design until we feel confident in our ability to get the job done. The point is to plan enough so that a lack of planning doesn’t create major problems later.

*Code Complete* is all about software construction so the planning McConnell writes about the most is related to the nitty-gritty: how to approach constructing classes and routines from variables, statements and control structures. This is not to say that other levels of planning such as requirements and architecture are less important, in fact, he spends the first part of the book talking about their importance and relation to construction activities. 

Let’s take a look at the nitty-gritty.

### The Pseudocode Programming Process
McConnell dedicated a whole chapter to this topic. The goal is to solve problems at the level of intent before jumping deep into implementation details. It’s often easier and therefore tempting to start writing code for a routine before clearly stating the problem it’s supposed to solve as well as all of the steps the routine will take. Blindly writing code is a gamble. You are betting your time (and therefore someone’s money) on the code you write to make it into production. This just increases the bond between you and the code which will make abandoning it—after you realize it won’t be needed—more difficult. Before making such bets, improve your chances with the PPP—the Pseudocode Programming Process.

The following may sound very obvious but bear with me for a few sentences. The goal is to think the problem through, identify steps to solve it, and as soon as you are sure that you can implement a certain step (or part of it) just write down a line of pseudocode with the intent of that step (or substep). This saves you the time of actually having to work out the details which is good, because you don’t yet know if this step will make it into production code.

How to Pseudocode:
-	Use English-like statements to precisely describe operations.
-	Make it as programming-language-independent as possible.
-	Keep it at a high enough level to justify its use. Write at the level of intent (what does the operation do instead of the specific steps to do it).
-	Keep it at a low enough level such that you feel comfortable converting it to production code.

The better and more familiar you are with the language you use and the problem you are solving, the [higher level]( https://i1.wp.com/usethebitcoin.com/wp-content/uploads/2017/11/BITCOIN_9000.jpg?w=618&ssl=1) your pseudocode tends to be. A beginner might have to write down the specific steps at first, but if he encounters the same problem multiple times, he will eventually chunk it into one line of pseudocode.

I often struggle with getting the granularity of pseudocode right. Sometimes I write pseudocode that’s detailed to the point where I might as well write code directly. Sometimes—on the other extreme—I write pseudocode on a level that’s too high—this leads me to gloss over problematic parts of the code I later try (and sometimes fail) to write.

Ideally, after converting the problem into actual code you will be able to reuse the pseudocode as comments—avoid redundant comments if the code is clear. This tends to improve readability which will make maintaining and reviewing your code easier.

Keep the above idea—thinking the problem through at the level of intent and only then fully committing to turning your solution into code—in mind while we next take a look at McConnell’s tips for constructing classes and routines.

### Tips for Constructing Classes:
1.	Create a general design for the class.
	-	Define the class’s responsibilities.
	-	Define what information the class will hide.
	-	Define exactly what abstraction the class interface will capture.
	-	Include the last three points as a comment in the source code if possible.
	-	Make sure that the class’s interface represents a consistent abstraction. (ex. If you offer a `findEmployee()` routine, it shouldn’t throw an `EOFException` but an `EmployeeNotFoundException`)
	- 	Determine whether the class will be derived from another class and whether other classes will be allowed to derive from it.
	-	Identify key public methods.
	-	Identify and design nontrivial data structures.
	-	Minimize accessibility, avoid exposing data and functionality when it’s not necessary to do so.
	-	Minimize coupling to other classes, avoid depending on code outside of the class as much as practically possible.
	-	Preserve integrity of the class’s interface and documentation as you modify it.

2.	Construct the routines within the class.
	-	Follow steps for constructing routines (see below).

3.	Review and test the class as a whole.
	-	Ideally, each routine is tested as it’s created. After the class starts taking shape it should be reviewed and tested as a whole in order to uncover any issues that can’t be tested at the individual routine level.

4.	Repeat if necessary.
	-	As most other processes in software engineering, this is by no means a linear process. For example, during construction of the individual routines (step 2), design errors—such as the need for additional routines—might become apparent. If so, go back to designing the class (step 1) before continuing with construction.
	-	Iterate until you are satisfied.

### Tips for Constructing Routines:
1.	Design the routine.
	- **Clearly** define the problem the routine is supposed to solve.
	- Name the routine such that the problem it solves is apparent.
	- Define information that the routine will hide.
	- Define inputs and outputs.
	- Define pre- and post-conditions (what is guaranteed to be true before and after the routine is called)
	- Think about efficiency but don’t sacrifice readability for dubious performance gains.
	- Research available algorithms and data structures, don’t reinvent wheels.
	- Summarize the routines job. Use the summary as a comment in the routines header. Ideally, the reader could treat the routine as a black box and only go into the implementation details if necessary.
	- Write the pseudocode (level of intent).

2.	Code the routine.
	-	Convert the pseudocode into actual code.
	-	Errors in the pseudocode might become more apparent while converting it to actual code. Expect to go back designing the routine (step 1) if you uncover serious errors that impact the whole routine.

3.	Review and test the code and design.
	-	Mentally check your routine for errors. 
	-	Does the pseudocode fully solve your problem?
	-	Does the code correspond to the pseudocode? 
	-	Step through your routine with a debugger. This step is so underrated. If you fully understand the routine you just wrote then it shouldn’t take much effort to go through it with a debugger.
	-	Test your routine.

4.	Repeat if necessary.
	-	Expect to heavily iterate over the above steps. You will often have to go into the details and implement some pseudocode to validate your approach, then you go back to the pseudocode, then back into implementation details and so on. Just make sure to minimize the time you spend with implementation details. Only implement things to support your reasoning on the pseudocode level, save time and avoid reasoning at the implementation level. 
	-	Iterate until you are satisfied.

Tips for testing routines:
-	Think about how you will test the routine, both before and as you write it. This tends to result in a modular design and often uncovers errors sooner.
-	Test all branches of your routine (ex. if you have a switch statement, test all cases).
-	Boundary analysis, test values +1, -1, and equal to boundaries to avoid off by one errors.
-	Dirty tests, check if your code fails when it should (too little/much data, invalid data, etc.)
-	Consider generating random inputs.
-	Ensure compatibility with old tests if available.

<br/>


I know, I know. Pre- and post-conditions? Pseudocode? Stepping through with a debugger? For every routine? The above tips sound tedious (they are) and your job is to ship code, that’s [fine]( https://blog.codinghorror.com/not-all-bugs-are-worth-fixing/). The above tips are suggestions to bring more structure into our thought process. Being aware of these optional steps and where they fit into our coding habits is in itself valuable.

## 2. Keep it simple

<p align="center">
  <img src="https://static1.squarespace.com/static/584579603e00be884523e4ed/t/5a087dc971c10b6a451acbeb/1510580829068/?format=2500w">
</p>

Lots of advice specific to writing nonfiction or writing code can be reduced to this: keep it simple.

> People at every level are prisoners of the notion that a simple style reflects a simple mind. Actually a simple style is the result of hard work and hard thinking; a muddled style reflects a muddled thinker or a person too arrogant, or too dumb, or too lazy to organize his thoughts.
>
> — <cite>William Zinsser</cite>  
<br/>


McConnell repeatedly writes that “managing complexity is Software’s Primary Technical Imperative”. At one point he refers to Fred Brook’s No Silver Bullets [paper]( http://worrydream.com/refs/Brooks-NoSilverBullet.pdf) which distinguishes two different types of complexity—essential and accidental. The point is that we should accept only as much complexity as necessary—the essential complexity of the problem at hand. Any rises in difficulty along the path to the final solution should be minimized. In a sense, all advice geared towards improving readability, modularity, maintainability and similar design goals is to increase understanding by reducing complexity. Note that in this post the word complexity refers to intellectual manageability, not computational complexity.

> Projects that fail for technical reasons mostly do so because the software is allowed to grow so complex that no one really knows what it does. When a project reaches the point at which no one completely understands the impact that code changes in one area will have on other areas, progress grinds to a halt.
>
> — <cite>Steve McConnell</cite>  
<br/>

So what can developers do to fight accidental complexity? 

Below I list some of the notes I took while reading *Code Complete*. Each bullet point is my attempt at summarizing a key idea from McConnell’s discussions. Depending on the amount of experience you have, some points will make more sense and some less. The only way to make the most of the advice is to go through the accompanying stories, studies, and code examples in the book. Nonetheless, I’m sure you will find something useful down there.

Treat the following list as a buffet, move on if something doesn’t seem interesting and feel free to pick up and adopt any suggestion you find useful.

Some of McConnell’s advice for reducing complexity:

* General
  * Before construction, make sure that the groundwork has been laid (problem is well defined, requirements are reasonably stable, architecture is sufficiently well defined, major risks have been addressed etc.)
  * „Hide complexity so that your brain doesn’t have to deal with it unless you’re specifically concerned with it. “ Be as restrictive as practically possible when it comes to visibility.
  * Limit the negative impact of changes by encapsulating areas that are likely to change such as business rules, hardware dependencies, input and output.
  * Make central points of control when possible (ex. put all code related to processing customer payments into one class or subsystem and keep it there). “The reduced-complexity benefit is that the fewer places you have to look for something the easier and safer it will be to change. “
  *	Assign responsibilities to everything—subsystems, classes, routines, variables, etc—this should help to justify their existence and clarify their usage.
  -	Use standard techniques whenever possible (widely known algorithms, data structures, design patterns, etc.)
  -	Consider using brute force. „A brute-force solution that works is better than an elegant solution that doesn’t work. “
  -	Minimize the amount of knowledge required to make a change. Push unnecessary details to another level so that you can think about them when you want to rather than thinking about all of the details all of the time.
  -	make code readable from top to bottom
  -	Quality gates. Set up checks during your project that determine if current work quality is good enough to continue working.

* Design
   - Form consistent abstractions. Keep the level of abstraction of public interfaces consistent. For example, if a class offers addEmployee() then it shouldn't offer nextItemInList() but nextEmployee(). Make sure to encapsulate data such that the resulting interface is consistent with the abstraction you want your class to represent.
   - Reduce coupling by keeping relations (between subsystems, classes, routines, etc.) small, direct, and flexible.
  -	Aim for strong cohesion. Code inside of a subsystem, or class, or routine should be strongly related and support a central purpose. (ex. avoid classes that encapsulate unrelated data or behavior). 
  -	Formalize pre- and post-conditions, what must be true in order to use X and what must be true after X finishes its job.
   -	„Design the interfaces so that changes are limited to the inside of the class and the outside remains unaffected. Any other class using the changed class should be unaware that the change has occurred. “
   -	Design classes with a high fan-in (ex. your class should be used by a lot of other classes such as a utility class), and a low-to-medium fan-out (ex. your class shouldn't use and depend on lots of other classes).
   -	Design for test. Designing such that testing is easy often results in formalized interfaces and decoupled subsystems which is generally beneficial.
   -	Keep your design modular. Draw diagrams and look at your code as a bunch of black boxes with well-defined interfaces.
   -	Experimental prototyping. Often you can’t fully define the design problem until you’ve at least partially solved it. In this context, prototyping means writing the absolute minimum amount of throwaway code that’s needed to answer a specific design question.
   -	Split up validation and work classes. On a class level you could designate the data validation to public classes and let private classes assume that the data they handle is clean. Code outside of the safety zone throws exceptions, code inside the safety zone uses assertions.

* Routines:
  -	Aim for functional cohesion, statements in a routine should all work together to accomplish exactly one job.
  -	Check pre- and post-conditions.
  -	Make the name of a routine as short or as long as necessary to make it understandable (other developers should know what the routine does by looking at its name).
  -	If possible, don’t exceed 200 lines of code, and use no more than 7 parameters.
  -	Consistently order parameters, group similar ones.
  -	Avoid using routine parameters as working variables, use local variables in your routine instead.
  -	Avoid passing parameters to store the output into, return the result.
  -	Document assumptions as you write the routine (you will forget them later).
  -	Keep [cyclomatic complexity](https://en.wikipedia.org/wiki/Cyclomatic_complexity) of your routines (number of paths, start with 1 and increment for every if, for, while, repeat, and, or, case...) below ~10.

* Loops:
  -	Comment loop headers, allow the reader to treat loops as black boxes.
  -	‎Put control structure related code at the beginning or end of the loop (ex. increment counter indices in a while loop at the beginning or end of its body)
  -	Loops should be short enough to be viewed all at once, aim for 15-20 lines.
  -	‎Loop depth should be at most 3, prepare a good explanation whenever exceeding this limit.
  -	‎Avoid altering the loop index in weird ways from inside of the loop.
  -	Avoid using the loop index outside of the loop
  -	‎Give the loop index, its initialization, and end condition meaningful names (avoid i, j, k, etc.).
  -	Using break or continue (or anything else inside of the loop that alters the control structure) increases the loop's complexity because the reader has to understand the loop’s body in order to understand the loop. Ideally, the reader should understand how the loop behaves by just looking at the contents of (say) a for loop’s parenthesis.
  -	Make entry and termination obvious, minimize the number of ways the loop can start and terminate.

* Conditionals:
  -	Outsource complicated tests to routines or variables. Ideally, the reader should understand what a complicated test checks without having to understand the implementation details.
  -	Show the normal path first, then exceptions. (ex. cover the most common cases first in else-if statements)
  -	Fully parenthesize expressions (how is a < b == c == d evaluated?) 
  -	‎Write numeric expressions in number line order, (min < i && i < max) instead of (max > i && min < i).


* Variables:
  -	Initialize constants at the beginning of a program, initialize member variables in constructors.
  -	Binding time. We differentiate coding time (magic numbers) < compile time (named constants) < load time (read from a file) < object instantiation time (read and set a variable upon object initialization) < just in time (read and set a value every time it is used).
After compile time, flexibility increases but so does complexity. The goal is to find a good trade-off based on the project's requirements.
  -	Make sure that every variable does exactly one job, never reuse a variable for uncorrelated purposes.
  -	Avoid implicit meanings (ex. special meaning when an integer variable is negative).
  -	Encapsulate primitive types in case you expect changes (ex. use a Weight class that internally uses doubles instead of just using doubles)
  -	Think about rounding errors, division by zero errors, overflows, avoid hard coding data.

* Developer Testing:
  -	Unit, component, integration, regression, system testing.
  -	Do clean and dirty tests, test if your code works but also test if your code fails when it should.
  -	Use coverage monitors to ensure high test coverage, choose a good metric such as branch coverage. Most developers are too optimistic when not using tools.

* Debugging:
  -	Understand the problem before trying to fix it.
  -	Add a unit test that triggers the error and keep it in order to prevent others/yourself from reintroducing the error.
  -	Don’t ignore compiler warnings, fix and understand all of them to avoid weird problems.

* Code Tuning:
  -	Best preparation for code tuning is writing clean, easy to understand and modifiable code.
  -	Performance relationships vary across languages, compilers, libraries, machines and versions. Mistrust any general claims about one technique being more efficient than another.
  -	Always use execution profilers to understand where your program spends its time.
  -	‎Only start tuning code if it works correctly. Measure bottlenecks, backup your working code, tune and measure the impact of every change.

* Comments:
  -	Avoid unnecessary comments that explain the obvious. Question the existence of each comment, delete if it’s not helpful (ex. `employees.addEmployee(employee); // adds employee`).
  -	Comment data declarations (intent, usage), blocks of code (intent), routines (inputs, outputs, assumptions, limitations, source of algorithms, global effects, source), loops (intent), class headers, etc.
  -	Only use commenting styles that are easy to maintain (ex. avoid fancy boxes and indentations).


* Collaborative Construction:
  -	[Formal inspections](https://en.wikipedia.org/wiki/Software_inspection), [pair programming](https://en.wikipedia.org/wiki/Pair_programming), walk-throughs, code readings.
  -	Studies show that formal inspections and pair programming are at least as effective as testing.
  -	Show your code to others, get feedback, and use it to improve not just your current work but your approach towards future problems.
  -	‎Do regression tests, they are essential to producing complex systems.
  -	Studies show that to maximize defect detection we should combine different testing techniques (formal/informal inspections, prototyping, developer testing, etc.) and views of different people during all stages (requirements, design, construction).

* Organizational:
  -	‎If you are behind, you likely won't catch up and it will only get worse. [Carefully](https://en.wikipedia.org/wiki/Brooks%27s_law) expand the team, reduce the scope of the project—focus on the most important parts, postpone deadlines.
  -	Don’t mindlessly apply changes that pop into your head. Store ideas (and requests) for change and deal with them systematically.
  -	Do daily builds (compile, link, produce an executable, get the code running), and smoke tests (thorough tests of the main features, doesn't have to be exhaustive).
  -	Check in code frequently and work in small increments to reduce the amount of integration errors.
  -	Make quality assurance a part of all development stages. Don’t postpone all testing till the end, it won’t be done properly.

* Miscellaneous
  - ‎Avoid recursion if it makes you feel uneasy. Only use it as a last resort when an iterative solution is very complex. Think about stack space when using recursion.
  - Avoid gotos. There are situations where their use is justified but use them as a last resort. They often hinder compiler optimizations, mistrust efficiency claims.
  - You should never have to look at the source code of some class in order to understand how to use it. You should be able to use a class by reading its documentation. Knowing a class’s implementation allows you to exploit it while using the class (often unconsciously). This is error-prone because the class' developers are only responsible to maintain the interface, implementation details that you assume might change and break your code.
  - Be aware of technology waves. Working with technology that is not mature yet means spending a large portion of the day trying to figure out how to use the technology (early wave). Most problems that you face will feel as if you are the first person experiencing them.
Working with mature technology means spending more time on building new functionality and less time on understanding the technology since most of the common tasks have been thought of and made easy by other programmers (late wave).
  - Use assertions to identify things that should NEVER happen, when an assertion goes off it's (ideally) not handled but the source code needs to be fixed. Use assertions to verify pre- and post-conditions or to check assumptions in safety zones (ex. private routines) such as checking the range of variables, state of a file/variable, object is not null, size of a data structure meets some criteria, etc.
  - „Establish programming conventions (naming, formatting, commenting, etc.) at the start of the project before you begin programming. It’s nearly impossible to change code to match them later. “ 

<br>

Most of this advice is there to keep developers from writing code that’s more complex than it has to be. Whenever I find myself leaning in a little too close towards the screen while working on some “smart” code, I try to lean back for a reality check. Does this have to be difficult? Am I just being silly and making things more difficult than they have to be? Am I using enough [hash maps]( https://i.redd.it/ym82hxxxq2d01.jpg)? More often than not I end up ditching the “smart” code and doing it the good old “boring” way.




## 3. Iterate, iterate, and iterate again

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/1000/1*gBLsSYp3M9gYhsgprRHVww.jpeg">
</p>

Books, articles, blogposts and non-trivial systems aren’t written in one go. Both authors emphasize the importance of heavily iterating over their work.

> Rewriting is the essence of writing well: it’s where the game is won or lost. That idea is hard to accept. We all have an emotional equity in our first draft; we can’t believe that it wasn’t born perfect. But the odds are close to 100 percent that it wasn’t. Most writers don’t initially say what they want to say, or say it as well as they could.
>
> — <cite>William Zinsser</cite>  
<br/>

The point is not to start with one approach and keep working on it till it’s good enough. The point is to acknowledge that mistakes will be made and learned from while making and abandoning attempts on a best effort basis. McConnell writes: “A first attempt might produce a solution that works, but it’s unlikely to produce the best solution. “

Fun [fact](https://arxiv.org/ftp/arxiv/papers/1702/1702.01715.pdf), Google rewrites most of their software every few years.

I’ll leave you with McConnell’s emphasis on the importance of an iterative process:



> Iteration is appropriate for many software-development activities. During your initial specification of a system, you work with the user through several versions of requirements until you’re sure you agree on them. That’s an iterative process. When you build flexibility into your process by building and delivering a system in several increments, that’s an iterative process. If you use prototyping to develop several alternative solutions quickly and cheaply before crafting the final product, that’s another form of iteration. Iterating on requirements is perhaps as important as any other aspect of the software-development process. Projects fail because they commit themselves to a solution before exploring alternatives. Iteration provides a way to learn about a product before you build it.
>
> — <cite>Steve McConnell</cite>  
<br/>

## Wind Up
I hope that you found some of the tips as useful as I did. Obviously, you won’t remember (and need) all of them, I tried summarizing the ones I think could help most developers. There is much more advice in the book. I would recommend *Code Complete*  to people that have programmed for about a year or two and would like to fill in gaps and get an overview of software construction.

To summarize, we talked about three main issues related to both, nonfiction writing and software construction. First we acknowledged the importance of clarifying thoughts and saw examples of how to structure the class and routine construction processes. Then we took a look at ~~eleventy~~ suggestions on how to keep it simple by avoiding accidental complexity. Finally, we underlined the importance of iterating over and over again until you are satisfied with the outcome.

In case you are interested in more books related to software engineering, McConnell provides a neat reading list at the end of the book. You can also find it [online](http://www.construx.com/Thought_Leadership/Books/Survival_Guide/Resources_By_Chapter/Recommended_Reading_Lists/).


