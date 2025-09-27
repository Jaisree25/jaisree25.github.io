---
layout: page
title: Notes & Posts
permalink: /posts/
---

<h2>All Posts</h2>

<ul>
{% for post in site.posts %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>  
    <span style="color: gray;">— {{ post.date | date: "%B %d, %Y" }}</span>
    {% if post.tags %}
      <br/>
      <small>
        Tags:
        {% for tag in post.tags %}
          <a href="#{{ tag | slugify }}">{{ tag }}</a>{% unless forloop.last %}, {% endunless %}
        {% endfor %}
      </small>
    {% endif %}
  </li>
{% endfor %}
</ul>

---

<h2>Browse by Tag</h2>

{% assign all_tags = site.tags | sort %}
{% for tag in all_tags %}
  <h3 id="{{ tag[0] | slugify }}">{{ tag[0] }}</h3>
  <ul>
    {% for post in tag[1] %}
      <li>
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>  
        <span style="color: gray;">— {{ post.date | date: "%B %d, %Y" }}</span>
      </li>
    {% endfor %}
  </ul>
{% endfor %}
