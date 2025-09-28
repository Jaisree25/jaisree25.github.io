---
layout: default
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
      <a class="btn btn-secondary" href="{{ '/competitions/' | relative_url }}">Competitions</a>
      <a class="btn btn-ghost" href="{{ '/posts/' | relative_url }}">Notes</a>
    </p>
  </div>
</div>

<!-- THREE CARDS -->
<div class="home-cards">
  <!-- Projects -->
  <a class="home-card" href="{{ '/projects/' | relative_url }}">
    <h3>🧪 Projects</h3>
    <p class="muted">
      {% assign items = site.projects %}
      {% if items and items.size > 0 %}
        {{ items.size }} project{% if items.size != 1 %}s{% endif %} • latest:
        {% assign latest = items | sort: 'year' | last %}
        {% if latest %} {{ latest.title }} {% if latest.year %}({{ latest.year }}){% endif %}{% endif %}
      {% else %}
        Add items under <code>_projects/</code>.
      {% endif %}
    </p>
  </a>

  <!-- Competitions -->
  <a class="home-card" href="{{ '/competitions/' | relative_url }}">
    <h3>🏆 Competitions</h3>
    <p class="muted">
      {% assign comps = site.competitions %}
      {% if comps and comps.size > 0 %}
        {{ comps.size }} entr{% if comps.size == 1 %}y{% else %}ies{% endif %} • latest:
        {% assign latestc = comps | sort: 'year' | last %}
        {% if latestc %} {{ latestc.title }} {% if latestc.year %}({{ latestc.year }}){% endif %}{% endif %}
      {% else %}
        Add items under <code>_competitions/</code>.
      {% endif %}
    </p>
  </a>

  <!-- Posts -->
  <a class="home-card" href="{{ '/posts/' | relative_url }}">
    <h3>📝 Notes & Posts</h3>
    <p class="muted">
      {% if site.posts and site.posts.size > 0 %}
        {{ site.posts.size }} post{% if site.posts.size != 1 %}s{% endif %} • latest:
        {{ site.posts[0].title }} ({{ site.posts[0].date | date: "%b %d, %Y" }})
      {% else %}
        Create files in <code>_posts/YYYY-MM-DD-title.md</code>.
      {% endif %}
    </p>
  </a>
</div>
