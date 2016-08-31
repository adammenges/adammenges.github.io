---
layout:     post
title:      Headline
date:       2014-12-14
summary:    When I was in college, I made a small script that would try and analyze the amount of time one spent looking for articles, and the amount of time one spent actually reading. To my surprise, for myself and friends, it's about half and half. That means we spend nearly *half* of our 'reading' time just looking for something good to read...
permalink: /the-philosophy-behind-headline/
---

When I was in college, I made a small script that would try and analyze the amount of time one spent looking for articles, and the amount of time one spent actually reading. To my surprise, for myself and friends, it's about half and half. That means we spend nearly *half* of our 'reading' time just looking for something good to read. I've noticed this elsewhere too, friends spend an enormous amount of time going through their netflix queue before actually finding something and watching it. Maybe not half and half, but closer then you'd think.

Unfortunately, we can't really decrease the amount of time it takes to absorb good material, Matrix style. Not yet anyway. But we can drastically decrease the amount of time it takes to find it.

I have a passion for automating aspects of my life prime for doing so. Headline is the product of doing just that. Daily, most of us the engineering world (not only software, but hardware, bio, mechanical, chemistry, etc.) stay up to date by logging onto a few news sites, browsing the top rated stuff, and filtering out the useless. In a world with click-bait, this is becoming harder and harder to do in a time efficient manner.

However, nothing about this isn't automatable. Let's break what we do into two steps.
    
- First, we gather top rated information. For example, well rated posts on hacker news / designer news, most upvoted posts on specific sub-reddits, or well re-tweeted / favorited tweets on twitter.
- Next, we look through them and mentally filter it down into relevant information, then read that.

Collecting the highly rated information isn't hard. Most of these services have APIs, and [BeautifulSoup](http://www.crummy.com/software/BeautifulSoup/bs4/doc/) was written for the rest of them. 😏

Next, filtering down to the most relevant information. This is where machine learning steps in. The goal of this part of the project is to give the users only information that they'd honestly be better off reading. This should not include an article about amazon releasing their new 'killer' phone, regardless of how much people are talking about it. Included, however, should be the release of [Feynman's lectures](feynmanlectures.caltech.edu) to the public.

> "All my work is based on the assumption that I am not different than the rest of mankind, therefore, if there is something in music that moves me, there's a good chance plenty of other people will experience the same."
> -- [Esa-Pekka Salonen](https://en.wikipedia.org/wiki/Esa-Pekka_Salonen)

Admittedly, these filters are tuned to my own personal preference. However, much like Esa-Pekka, I believe there is value in this. Maybe in the future these filters could be more individually tuned, though I'd need to figure out some way to support feedback through pocket.

Perhaps I'm abusing pocket. I don't really add anything to my list using the typical workflow (find a article, don't have time to read it now, put it in my 'read it later' list). But in a world with so many read it later services, where they are no longer new, pocket needs to evolve, or risk becoming irrelevant.

Pocket is now the perfect system for me, Headline adds filtered popular content, put in combination with some IFTTT recipes, it also automatically gets posts from blogs/authors I like. All this without the cruft, just a clean, synced, beautiful interface. I no longer spend time finding good content, I simply open the pocket app and start reading.

I get the benefits, without the timesink. [Check it out.](http://headline.adammenges.com/)