---
layout: default
title: "Home"
---

<!-- HERO -->
<div class="hero">
  <div class="hero-banner-wrap">
    <img class="hero-banner" src="{{ '/assets/images/banner.png' | relative_url }}" alt="Jaisree banner" />
    <img src="{{ '/assets/images/j.ai_cropped.png' | relative_url }}" alt="J.ai logo" class="hero-avatar" />
  </div>
  <div class="hero-copy">
    <h1>AI • Robotics • Intelligent Systems</h1>
    <p>An aspiring AI engineer passionate about building and writing about machine learning, robotics, and autonomous systems.</p>
  </div>
</div>

<!-- LATEST POSTS (3 cards) -->
<h2>Latest Posts</h2>
<div class="post-cards">
  {% if site.posts and site.posts.size > 0 %}
    {% for post in site.posts limit:3 %}
      <a class="post-card" href="{{ post.url | relative_url }}">
        <h3 class="post-card__title">{{ post.title }}</h3>
        <div class="post-card__meta">{{ post.date | date: "%b %d, %Y" }}{% if post.tags and post.tags.size > 0 %} • {{ post.tags | join: ", " }}{% endif %}</div>
        <p class="post-card__excerpt">
          {%- if post.excerpt -%}
            {{ post.excerpt | strip_html | truncate: 160 }}
          {%- else -%}
            {{ post.content | strip_html | truncate: 160 }}
          {%- endif -%}
        </p>
        <span class="post-card__more">Read more →</span>
      </a>
    {% endfor %}
  {% else %}
    <p>No posts yet. Add files under <code>_posts/YYYY-MM-DD-title.md</code>.</p>
  {% endif %}
</div>

<p><a class="link-more" href="{{ '/posts/' | relative_url }}">See all posts →</a></p>
