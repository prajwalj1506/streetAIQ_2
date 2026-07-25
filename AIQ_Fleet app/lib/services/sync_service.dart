import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'database_service.dart';

class SyncService extends ChangeNotifier {
  static final SyncService _instance = SyncService._internal();
  factory SyncService() => _instance;
  SyncService._internal();

  Timer? _syncTimer;
  bool _isSyncing = false;
  bool _isServerOnline = false;

  bool get isServerOnline => _isServerOnline;

  void startSyncTimer() {
    _syncTimer = Timer.periodic(const Duration(seconds: 10), (timer) {
      _syncData();
    });
  }

  Future<void> _syncData() async {
    if (_isSyncing) return;
    _isSyncing = true;

    try {
      if (kIsWeb) {
        // Skip direct HTTP sync on web to avoid CORS errors with ngrok.
        // Firebase Firestore already handles live tracking.
        return;
      }

      final prefs = await SharedPreferences.getInstance();
      final backendUrl = prefs.getString('backendUrl') ?? 'https://frankfurt-tour-chef-antibody.trycloudflare.com';
      final vehicleId = prefs.getString('vehicleId') ?? 'V001';

      // 1. Sync Telemetry
      final telemetry = await DatabaseService().getUnsyncedTelemetry();
      if (telemetry.isNotEmpty) {
        final response = await http.post(
          Uri.parse('$backendUrl/api/v1/telemetry'),
          headers: {'Content-Type': 'application/json', 'X-Vehicle-ID': vehicleId},
          body: json.encode({'batch': telemetry}),
        ).timeout(const Duration(seconds: 5));

        if (response.statusCode == 200 || response.statusCode == 201) {
          _isServerOnline = true;
          for (var item in telemetry) {
            await DatabaseService().deleteTelemetry(item['id']);
          }
        } else {
          _isServerOnline = false;
        }
      }

      // 2. Sync Video Chunks (Mock Implementation for now)
      // Usually requires multipart form upload
      final chunks = await DatabaseService().getUnsyncedVideoChunks();
      if (chunks.isNotEmpty && _isServerOnline) {
        // Mock success
        for (var item in chunks) {
          await DatabaseService().deleteVideoChunk(item['id']);
        }
      }

      notifyListeners();
    } catch (e) {
      _isServerOnline = false;
      notifyListeners();
      debugPrint("Sync Error: $e");
    } finally {
      _isSyncing = false;
    }
  }

  void disposeService() {
    _syncTimer?.cancel();
  }
}
