---
layout: page
title: Posts
permalink: /posts/
---

<h2>All Posts</h2>

<ul>
{% for post in site.posts %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>  
    <span style="color: gray;">— {{ post.date | date: "%B %d, %Y" }}</span>
  </li>
{% endfor %}
</ul>

