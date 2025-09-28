---
layout: page
title: Posts
permalink: /posts/
---

<div class="post-cards">
{% if site.posts and site.posts.size > 0 %}
  {% for post in site.posts %}
    <a class="post-card" href="{{ post.url | relative_url }}">
      <h3 class="post-card__title">{{ post.title }}</h3>
      <div class="post-card__meta">
        {{ post.date | date: "%b %d, %Y" }}
        {% if post.tags and post.tags.size > 0 %}
          • {{ post.tags | join: ", " }}
        {% endif %}
      </div>
      <p class="post-card__excerpt">
        {%- if post.excerpt -%}
          {{ post.excerpt | strip_html | truncate: 180 }}
        {%- else -%}
          {{ post.content | strip_html | truncate: 180 }}
        {%- endif -%}
      </p>
      <span class="post-card__more">Read more →</span>
    </a>
  {% endfor %}
{% else %}
  <p>No posts yet. Create files under <code>_posts/YYYY-MM-DD-title.md</code>.</p>
{% endif %}
</div>
