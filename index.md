---
layout: home
title: "Jaisree’s — Research"
---

<!-- HERO -->
<div class="hero">
  <img class="hero-avatar" src="{{ '/assets/images/profile.webp' | relative_url }}" alt="Jaisree headshot" width="120" height="120" />
  <div class="hero-copy">
    <h1>AI • Robotics • Systems</h1>
    <p>I build and write about embodied AI, robot learning, and reliable systems.</p>
    <p class="hero-ctas">
      <a class="btn" href="{{ '/projects/' | relative_url }}">Projects</a>
      <a class="btn btn-secondary" href="{{ '/publications/' | relative_url }}">Publications</a>
      <a class="btn btn-ghost" href="{{ '/posts/' | relative_url }}">Notes</a>
    </p>
  </div>
</div>

<!-- FEATURED PROJECTS -->
<h2>Featured Projects</h2>
<div class="card-grid">
  {% assign feats = site.projects | where: 'featured', true | sort: 'year' | reverse | slice: 0,3 %}
  {% if feats.size == 0 %}
    <p>Add <code>featured: true</code> to 1–3 files under <code>_projects/</code> to populate this section.</p>
  {% endif %}
  {% for p in feats %}
    <a class="card" href="{{ p.url | relative_url }}">
      <h3>{{ p.title }}</h3>
      <div class="meta">
        {% if p.year %}{{ p.year }}{% endif %}
        {% if p.tags %} • {{ p.tags | join: ', ' }}{% endif %}
      </div>
      {% if p.summary %}<p>{{ p.summary }}</p>{% endif %}
    </a>
  {% endfor %}
</div>

<!-- LATEST POSTS -->
<h2>Latest Posts</h2>
<ul class="post-list">
  {% for post in site.posts limit:3 %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <span class="muted">— {{ post.date | date: "%b %d, %Y" }}</span>
      {% if post.excerpt %}<div class="excerpt">{{ post.excerpt | strip_html }}</div>{% endif %}
    </li>
  {% endfor %}
</ul>

<p><a class="link-more" href="{{ '/posts/' | relative_url }}">Browse all posts →</a></p>
