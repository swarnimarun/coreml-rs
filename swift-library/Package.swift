// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "swift-library",
    platforms: [.macOS(.v13)],
    products: [
        .library(
            name: "swift-library",
            type: .static,
            targets: ["swift-library"])
    ],
    targets: [
        .target(
            name: "swift-library")
    ]
)
