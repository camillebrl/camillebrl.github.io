// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-publications",
          title: "publications",
          description: "you can find here all my publications!",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-projects",
          title: "projects",
          description: "Detail of my personal projects.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-repositories",
          title: "repositories",
          description: "Here are some stats about my personnal github and projects.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "You can find here my CV.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "dropdown-bookshelf",
              title: "bookshelf",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/books/";
              },
            },{id: "dropdown-blog",
              title: "blog",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "/blog/";
              },
            },{id: "post-tips-mathématiques-points-utiles-en-ia",
        
          title: "Tips mathématiques / points utiles en IA",
        
        description: "Tips mathématiques / points utiles en IA",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/tips_mathematics_for_ai/";
          
        },
      },{id: "post-les-fonctions-d-39-influence-appliquées-aux-llms",
        
          title: "Les fonctions d&#39;influence appliquées aux LLMs",
        
        description: "Détail de ma compréhension des fonctions d&#39;influence appliquées aux LLMs",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/influence_functions_applied_to_llms/";
          
        },
      },{id: "post-les-approches-d-39-explicabilité-des-llms",
        
          title: "Les approches d&#39;explicabilité des LLMs",
        
        description: "Méthodes d&#39;explicabilité de la génération de texte par les LLMs",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/explainability_llm_generation/";
          
        },
      },{id: "post-biais-positionnels-dans-les-transformers-auto-régressifs",
        
          title: "Biais Positionnels dans les transformers auto-régressifs",
        
        description: "Description du biais positionnel dans les transformers auto-régressifs",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/positional_biais/";
          
        },
      },{id: "books-foundations-of-computer-vision",
          title: 'Foundations of Computer Vision',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/foundations_cv/";
            },},{id: "books-an-introduction-to-statistical-learning-with-applications-in-python",
          title: 'An Introduction to Statistical Learning with Applications in Python',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/statistical_learning/";
            },},{id: "projects-doccustomkie",
          title: 'DocCustomKIE',
          description: "An end-to-end pipeline for custom Key Information Extraction from documents",
          section: "Projects",handler: () => {
              window.location.href = "/projects/1_project/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%63%61%6D%69%6C%6C%65.%62%61%72%62%6F%75%6C%65@%67%6D%61%69%6C.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/camillebrl", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/camille-barboule-353829147", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=pDIkfAEAAAAJ", "_blank");
        },
      },{
        id: 'social-semanticscholar',
        title: 'Semantic Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://www.semanticscholar.org/author/2336738897", "_blank");
        },
      },{
        id: 'social-x',
        title: 'X',
        section: 'Socials',
        handler: () => {
          window.open("https://twitter.com/BarbouleCamille", "_blank");
        },
      },{
        id: 'social-orcid',
        title: 'ORCID',
        section: 'Socials',
        handler: () => {
          window.open("https://orcid.org/0009-0001-9182-7120", "_blank");
        },
      },{
        id: 'social-zotero',
        title: 'Zotero',
        section: 'Socials',
        handler: () => {
          window.open("https://www.zotero.org/cambrl", "_blank");
        },
      },{
        id: 'social-bluesky',
        title: 'Bluesky',
        section: 'Socials',
        handler: () => {
          window.open("https://bsky.app/profile/camillebarboule.bsky.social", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
