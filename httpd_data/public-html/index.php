<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sample Website</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <header>
        <nav class="navbar">
            <div class="container">
                <h1 class="logo">My Website</h1>
                <ul class="nav-links">
                    <li><a href="#home">Home</a></li>
                    <li><a href="#about">About</a></li>
                    <li><a href="#services">Services</a></li>
                    <li><a href="#user">Users</a></li>
                </ul>
                <!-- Camouflaged Search Bar -->
                <form id="searchForm" class="search-bar" action="" method="GET">
                    <input type="text" id="searchInput" name="search" placeholder="Search..." class="search-input">
                    <button type="submit" class="search-btn">Search</button>
                </form>
            </div>
        </nav>
    </header>

    <main>
        <section id="home" class="hero">
            <div class="container">
                <h2>Welcome to My Website</h2>
                <p>Discover content and services tailored for you.</p>
                <a href="#about" class="btn">Learn More</a>
                <!-- Placeholder for search results -->
                <div id="searchResult" class="search-result"></div>
            </div>
        </section>

        <section id="about" class="content">
            <div class="container">
                <h2>About Us</h2>
                <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Proin nec lorem a neque venenatis fermentum.</p>
            </div>
        </section>

        <section id="services" class="content">
            <div class="container">
                <h2>Our Services</h2>
                <p>Explore the variety of services we offer to meet your needs.</p>
            </div>
        </section>

        <section id="user" class="content">
            <div class="container">
                <h2>Users</h2>
                <form id="userForm" class="form">
                    <label for="userId">User ID:</label>
                    <input type="text" id="userId" name="userId" class="input" placeholder="Enter user ID">
                    <br>
                    <button type="button" id="fetchUser" class="btn">Fetch User</button>
                </form>
                <div id="userResult" class="result">
                    <!-- Results from the microservice will appear here -->
                </div>
            </div>
        </section>
    </main>

    <footer>
        <div class="container">
            <p>&copy; 2024 My Website. All rights reserved.</p>
        </div>
    </footer>

    <script>
        // Fetch user data from the microservice
        document.getElementById("fetchUser").addEventListener("click", () => {
            const userId = document.getElementById("userId").value;
            const userResult = document.getElementById("userResult");

            fetch(`http://localhost:3000/user?id=${userId}`)
                .then((response) => response.json())
                .then((data) => {
                    console.log(data);
                    if (data.length > 0) {
                        userResult.innerHTML = `<p>User found: ${JSON.stringify(data)}</p>`;
                    } else {
                        userResult.innerHTML = `<p>No user found with ID ${userId}.</p>`;
                    }
                })
                .catch((error) => {
                    userResult.innerHTML = `<p>Error fetching user data: ${error.message}</p>`;
                });
        });
        // Function to get URL parameters
        function getQueryParam(param) {
            const urlParams = new URLSearchParams(window.location.search);
            return urlParams.get(param);
        }

        // On page load, process the 'search' parameter if it exists
        document.addEventListener("DOMContentLoaded", () => {
            const searchResult = document.getElementById("searchResult");

            // Get the 'search' parameter from the URL
            const searchParam = getQueryParam("search");

            if (searchParam) {
                // Vulnerable XSS: Directly inject user input into the DOM
                searchResult.innerHTML = `<p>Search result for: ${searchParam}</p>`;
            }
        });

        // Attach the submit event listener to the search form
        document.getElementById("searchForm").addEventListener("submit", (event) => {
            // Prevent default form submission behavior
            event.preventDefault();

            // Redirect to the same page with the search parameter in the URL
            const searchInput = document.getElementById("searchInput").value;
            window.location.href = `?search=${encodeURIComponent(searchInput)}`;
        });
    </script>
</body>
</html>
