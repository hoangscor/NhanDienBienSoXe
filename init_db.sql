CREATE DATABASE IF NOT EXISTS parking_system
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE parking_system;

CREATE TABLE IF NOT EXISTS parking_log (
  id INT AUTO_INCREMENT PRIMARY KEY,
  plate VARCHAR(20),
  entry_time DATETIME,
  exit_time DATETIME,
  fee INT
);
