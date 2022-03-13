push!(LOAD_PATH, "../src/")

using Documenter, IterativeLQR

makedocs(
    modules = [IterativeLQR],
    format = Documenter.HTML(prettyurls=false),
    sitename = "IterativeLQR",
    pages = [
        ##############################################
        ## MAKE SURE TO SYNC WITH docs/src/index.md ##
        ##############################################
        "index.md",

        "faq.md",
        "api.md",
        "contributing.md",
        "citing.md"
    ]
)

deploydocs(
    repo = "github.com/thowell/IterativeLQR.jl.git",
)
